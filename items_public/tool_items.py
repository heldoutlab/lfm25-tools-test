# -*- coding: utf-8 -*-
"""Mock tool environments + long multi-step tool-calling tasks (lfm25-tools PUBLIC task set).

Stdlib only; runs on Python 3.9 and 3.12.  Tasks are graded on the FINAL STATE
of the mock environment (task["check"](env.state)).

Layout
  ENVS   name -> environment class (TOOLS in OpenAI chat-completions format,
         __init__(init), call(name, args) which never raises and leaves the
         state unchanged whenever it returns {"error": ...})
  TASKS  T01..T75 (test set, 15 per environment: T01-T40 first round, T41-T75
         extension) then TP1..TP6 (calibration, same style and difficulty; one
         per environment plus one extra)

Domains (all new relative to the bonsai2 rounds): public-library holds and
inter-branch transit, IT asset and licence management, meal-kit production
planning with allergens and stock lots, a content-moderation queue with an
escalation policy, and exam invigilation.

  * references need 7-16 calls; rules live in policy read-tools that the write tools enforce
  * no schema uses a union type and no task needs null / "" as an argument
  * every entity a prompt names is mechanically checked to exist in init
  * "today" comes from a read tool, never from the real clock

Run this file directly to self-verify:  prints "OK 81/81".
"""
import copy
import datetime
import itertools
import json
import random
import re


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------
class _Rule(Exception):
    """A business-rule / lookup failure; turned into {"error": ...}."""


def _norm(schema, v, path):
    """Validate v against a (small subset of) JSON Schema. Returns (value, err)."""
    t = schema.get("type")
    if v is None:
        return None, "'%s' must not be null" % path
    if t == "string":
        if not isinstance(v, str):
            return None, "'%s' must be a string" % path
        if "enum" in schema and v not in schema["enum"]:
            return None, "'%s' must be one of: %s" % (path, ", ".join(schema["enum"]))
        return v, None
    if t == "integer":
        if isinstance(v, bool):
            return None, "'%s' must be an integer" % path
        if isinstance(v, str) and re.match(r"^-?\d{1,9}$", v.strip()):
            v = int(v.strip())   # "250" for 250 is a tool-call formatting slip, not a reasoning error
        if isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")) or v != int(v):
                return None, "'%s' must be an integer" % path
            v = int(v)
        if not isinstance(v, int):
            return None, "'%s' must be an integer" % path
        if abs(v) > 10 ** 9:
            return None, "'%s' is out of range" % path
        if "minimum" in schema and v < schema["minimum"]:
            return None, "'%s' must be >= %d" % (path, schema["minimum"])
        if "maximum" in schema and v > schema["maximum"]:
            return None, "'%s' must be <= %d" % (path, schema["maximum"])
        return v, None
    if t == "boolean":
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            v = v.strip().lower() == "true"   # same leniency for "true"/"false" sent as strings
        if not isinstance(v, bool):
            return None, "'%s' must be true or false" % path
        return v, None
    if t == "array":
        if not isinstance(v, list):
            return None, "'%s' must be an array" % path
        out = []
        for n, item in enumerate(v):
            val, err = _norm(schema["items"], item, "%s[%d]" % (path, n))
            if err:
                return None, err
            out.append(val)
        return out, None
    if t == "object":
        if not isinstance(v, dict):
            return None, "'%s' must be an object" % path
        props = schema.get("properties", {})
        required = schema.get("required", [])
        pre = "" if path == "arguments" else path + "."
        out = {}
        for k in v:
            if not isinstance(k, str) or k not in props:
                return None, "unexpected argument '%s%s' (allowed: %s)" % (
                    pre, str(k)[:40], ", ".join(sorted(props)) or "none")
        for k in required:
            if k not in v:
                return None, "missing required argument '%s%s'" % (pre, k)
        for k, val in v.items():
            if val is None and k not in required:
                continue  # optional argument sent as null == omitted
            val, err = _norm(props[k], val, pre + k)
            if err:
                return None, err
            out[k] = val
        return out, None
    return None, "'%s' has an unsupported type" % path


def _fn(name, desc, props=None, required=()):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {},
                       "required": list(required), "additionalProperties": False}}}


def _s(desc, enum=None):
    d = {"type": "string", "description": desc}
    if enum:
        d["enum"] = list(enum)
    return d


def _i(desc, minimum=None, maximum=None):
    d = {"type": "integer", "description": desc}
    if minimum is not None:
        d["minimum"] = minimum
    if maximum is not None:
        d["maximum"] = maximum
    return d


def _b(desc):
    return {"type": "boolean", "description": desc}


_PAGE = _i("Page number, starting at 1 (default 1).", 1)


def _pdate(s, what="date"):
    if not isinstance(s, str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        raise _Rule("%s must be formatted YYYY-MM-DD, got '%s'" % (what, str(s)[:30]))
    try:
        return datetime.date(int(s[0:4]), int(s[5:7]), int(s[8:10]))
    except ValueError:
        raise _Rule("%s '%s' is not a real calendar date" % (what, s))


def _ptime(s, what="time"):
    if not isinstance(s, str) or not re.match(r"^\d{2}:\d{2}$", s):
        raise _Rule("%s must be formatted HH:MM (24h), got '%s'" % (what, str(s)[:30]))
    h, m = int(s[0:2]), int(s[3:5])
    if h > 23 or m > 59:
        raise _Rule("%s '%s' is not a valid time" % (what, s))
    return h * 60 + m


def _fmt_time(minutes):
    return "%02d:%02d" % (minutes // 60, minutes % 60)


def _plus_days(s, n):
    return (_pdate(s) + datetime.timedelta(days=n)).isoformat()


def _paged(rows, page, size):
    """Returns (meta, slice). Raises _Rule for a page that does not exist."""
    pages = max(1, (len(rows) + size - 1) // size)
    if page > pages:
        raise _Rule("page %d does not exist (there are %d page(s))" % (page, pages))
    return ({"page": page, "pages": pages, "total_results": len(rows)},
            rows[(page - 1) * size: page * size])


_DAYNAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_READ_PREFIXES = ("get_", "list_", "find_", "search_")


def _is_read(name):
    return isinstance(name, str) and name.startswith(_READ_PREFIXES)


def _clone(state):
    """Deep copy of a JSON-shaped state (much faster than copy.deepcopy)."""
    return json.loads(json.dumps(state))


class _Env(object):
    TOOLS = []

    def __init__(self, init):
        self.state = _clone(init)

    def _restore(self, snap):
        self.state.clear()
        self.state.update(snap)

    def call(self, name, args):
        try:
            return self._call(name, args)
        except Exception as e:  # never raise
            return {"error": "invalid call (%s)" % type(e).__name__}

    def _call(self, name, args):
        if not isinstance(name, str):
            return {"error": "tool name must be a string"}
        spec = None
        for t in self.TOOLS:
            if t["function"]["name"] == name:
                spec = t["function"]
        if spec is None:
            return {"error": "unknown tool '%s'; available tools: %s" % (
                name[:60], ", ".join(t["function"]["name"] for t in self.TOOLS))}
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {"error": "arguments must be a JSON object"}
        args, err = _norm(spec["parameters"], args, "arguments")
        if err:
            return {"error": err}
        snap = _clone(self.state)
        try:
            res = getattr(self, "_t_" + name)(args)
            res = json.loads(json.dumps(res))
            json.dumps(self.state)
        except _Rule as e:
            self._restore(snap)
            return {"error": str(e)}
        except Exception as e:
            self._restore(snap)
            return {"error": "internal error (%s); nothing was changed" % type(e).__name__}
        if isinstance(res, dict) and "error" in res:
            self._restore(snap)
        return res

    def _get(self, coll, key, what):
        rec = self.state[coll].get(key)
        if rec is None:
            raise _Rule("unknown %s '%s'" % (what, key[:40]))
        return rec

    def _new_id(self, counter, prefix):
        n = self.state[counter]
        self.state[counter] = n + 1
        return "%s-%d" % (prefix, n)


# --------------------------------------------------------------------------
# 1. library: branches, titles, copies, patrons, holds queue, transit
# --------------------------------------------------------------------------
_COPY_STATUS = ["on_shelf", "on_loan", "on_hold_shelf", "in_transit", "missing", "withdrawn"]
_HOLD_STATUS = ["waiting", "suspended", "in_transit", "ready", "fulfilled", "cancelled"]
_ACTIVE_HOLD = ("waiting", "suspended", "in_transit", "ready")


class LibraryEnv(_Env):
    TOOLS = [
        _fn("get_library_info", "Today's date, the branches (with open/closed status) and the circulation rules "
            "that the write tools enforce."),
        _fn("find_patrons", "Search patrons by name (case-insensitive substring). Returns patron ids, type, home "
            "branch and card issue date.",
            {"query": _s("Text that must appear in the patron name.")}, ["query"]),
        _fn("get_patron", "One patron: type, home branch, blocked flag, hold limit, number of active holds and "
            "current loans.",
            {"patron_id": _s("Patron id, e.g. P-201.")}, ["patron_id"]),
        _fn("search_titles", "Search the catalogue by title or author (case-insensitive substring).",
            {"query": _s("Text that must appear in the title or author."),
             "format": _s("Only this format.", ["book", "large_print", "audiobook", "dvd"])}, ["query"]),
        _fn("list_copies", "Physical copies, filtered; 8 per page, by copy id. Shows location, status, condition, "
            "transit destination and attached hold.",
            {"title_id": _s("Only copies of this title, e.g. BK-110."),
             "location": _s("Only copies currently at (or, in transit, coming from) this branch code."),
             "status": _s("Only copies in this status.", _COPY_STATUS),
             "page": _PAGE}),
        _fn("get_copy", "One copy in detail.", {"copy_id": _s("Copy id, e.g. CP-1101.")}, ["copy_id"]),
        _fn("list_holds", "Holds, filtered, in queue order (date placed, then hold id); 6 per page.",
            {"title_id": _s("Only holds on this title."), "patron_id": _s("Only holds of this patron."),
             "pickup_branch": _s("Only holds with this pickup branch code."),
             "status": _s("Only holds in this status.", _HOLD_STATUS), "page": _PAGE}),
        _fn("place_hold", "Place a new hold for a patron (it joins the end of the title's queue as 'waiting').",
            {"patron_id": _s("Patron id."), "title_id": _s("Title id."),
             "pickup_branch": _s("Branch code where the patron will collect it.")},
            ["patron_id", "title_id", "pickup_branch"]),
        _fn("cancel_hold", "Cancel a waiting, suspended or ready hold. A copy waiting on the hold shelf for it "
            "goes back on the open shelf at that branch.",
            {"hold_id": _s("Hold id, e.g. HD-501.")}, ["hold_id"]),
        _fn("fill_hold", "Give a copy to a waiting hold. If the copy is at the hold's pickup branch the hold becomes "
            "ready at once; otherwise the copy is sent in transit to the pickup branch.",
            {"hold_id": _s("Hold id."), "copy_id": _s("Copy id.")}, ["hold_id", "copy_id"]),
        _fn("receive_copy", "Check in a copy that has arrived from transit at its destination branch. If a hold is "
            "attached, the hold becomes ready; otherwise the copy goes on the open shelf.",
            {"copy_id": _s("Copy id.")}, ["copy_id"]),
        _fn("update_hold", "Change the pickup branch of a hold.",
            {"hold_id": _s("Hold id."), "pickup_branch": _s("New pickup branch code.")}, ["hold_id", "pickup_branch"]),
        _fn("set_hold_suspended", "Suspend a waiting hold (suspended=true) or reactivate a suspended one "
            "(suspended=false). The hold keeps its date placed.",
            {"hold_id": _s("Hold id."), "suspended": _b("true to suspend, false to reactivate.")},
            ["hold_id", "suspended"]),
        _fn("set_copy_status", "Mark an open-shelf copy missing or withdrawn, or put a missing copy back on the shelf.",
            {"copy_id": _s("Copy id."), "status": _s("New status.", ["missing", "withdrawn", "on_shelf"])},
            ["copy_id", "status"]),
    ]

    def _branch(self, code, must_open=False):
        b = self.state["branches"].get(code)
        if b is None:
            raise _Rule("unknown branch code '%s' (codes: %s)" % (code[:20], ", ".join(sorted(self.state["branches"]))))
        if must_open and b["status"] != "open":
            raise _Rule("branch %s (%s) is closed until %s" % (code, b["name"], b["reopens"]))
        return b

    def _active(self, pid):
        return [k for k, h in sorted(self.state["holds"].items()) if h["patron"] == pid and h["status"] in _ACTIVE_HOLD]

    def _limit(self, pid):
        return self.state["policy"]["max_active_holds"][self.state["patrons"][pid]["type"]]

    def _eligible(self, h):
        p = self.state["patrons"][h["patron"]]
        return h["status"] == "waiting" and not p["blocked"] and \
            self.state["branches"][h["pickup"]]["status"] == "open"

    def _queue(self, title):
        return sorted((h["placed"], k) for k, h in self.state["holds"].items() if h["title"] == title)

    def _hold_row(self, k):
        h = self.state["holds"][k]
        return {"hold_id": k, "title_id": h["title"], "title": self.state["titles"][h["title"]]["title"],
                "format": self.state["titles"][h["title"]]["format"], "patron_id": h["patron"],
                "patron_name": self.state["patrons"][h["patron"]]["name"], "pickup_branch": h["pickup"],
                "status": h["status"], "placed": h["placed"], "copy_id": h["copy"], "ready_until": h["ready_until"]}

    def _copy_row(self, k):
        c = self.state["copies"][k]
        return {"copy_id": k, "title_id": c["title"], "title": self.state["titles"][c["title"]]["title"],
                "location": c["location"], "status": c["status"], "condition": c["condition"],
                "destination": c["destination"], "hold_id": c["hold"], "due": c["due"]}

    def _t_get_library_info(self, a):
        p = self.state["policy"]
        return {"today": self.state["today"], "branches": [
            dict(code=k, **v) for k, v in sorted(self.state["branches"].items())],
            "policy": {
                "hold_shelf_days": p["hold_shelf_days"], "max_active_holds": p["max_active_holds"],
                "rules": [
                    "Active holds are those waiting, suspended, in transit or ready. A patron may have at most "
                    "max_active_holds active holds (by patron type) and only one active hold per title.",
                    "A patron cannot place a hold on a title of which they currently have a copy on loan.",
                    "Blocked patrons cannot place holds, and their existing holds are passed over when copies are "
                    "handed out (they keep their place in the queue).",
                    "Queue order: a copy may only fill the OLDEST eligible waiting hold on its title. Eligible means "
                    "status waiting, patron not blocked, and pickup branch open. Suspended holds are passed over.",
                    "Only copies that are on the open shelf, in good or worn condition (never damaged), at an open "
                    "branch can fill a hold or be sent anywhere.",
                    "A ready hold stays on the hold shelf until its ready_until date (inclusive): hold_shelf_days "
                    "after it became ready. After that date it has expired and may be cancelled.",
                    "The pickup branch can only be changed on a hold that is waiting; a suspended hold must be "
                    "reactivated first. Holds that are in transit or ready cannot be redirected. New pickup "
                    "branches must be open.",
                    "Holds in transit cannot be cancelled until the copy has been received.",
                    "Copies can be marked missing or withdrawn only while on the open shelf (or, for withdrawn, "
                    "while missing).",
                ]}}

    def _t_find_patrons(self, a):
        q = a["query"].strip().lower()
        if not q:
            raise _Rule("query must not be empty")
        return {"patrons": [{"patron_id": k, "name": p["name"], "type": p["type"], "home_branch": p["home"],
                             "card_issued": p["card_issued"]}
                            for k, p in sorted(self.state["patrons"].items()) if q in p["name"].lower()]}

    def _t_get_patron(self, a):
        pid = a["patron_id"]
        p = self._get("patrons", pid, "patron")
        loans = [{"copy_id": k, "title_id": c["title"], "title": self.state["titles"][c["title"]]["title"],
                  "due": c["due"]} for k, c in sorted(self.state["copies"].items())
                 if c["status"] == "on_loan" and c["borrower"] == pid]
        return {"patron_id": pid, "name": p["name"], "type": p["type"], "home_branch": p["home"],
                "card_issued": p["card_issued"], "blocked": p["blocked"], "hold_limit": self._limit(pid),
                "active_holds": len(self._active(pid)), "loans": loans}

    def _t_search_titles(self, a):
        q = a["query"].strip().lower()
        if not q:
            raise _Rule("query must not be empty")
        return {"titles": [dict(title_id=k, **t) for k, t in sorted(self.state["titles"].items())
                           if (q in t["title"].lower() or q in t["author"].lower())
                           and ("format" not in a or t["format"] == a["format"])]}

    def _t_list_copies(self, a):
        if "title_id" in a:
            self._get("titles", a["title_id"], "title")
        if "location" in a:
            self._branch(a["location"])
        rows = [k for k, c in sorted(self.state["copies"].items())
                if ("title_id" not in a or c["title"] == a["title_id"])
                and ("location" not in a or c["location"] == a["location"])
                and ("status" not in a or c["status"] == a["status"])]
        meta, rows = _paged(rows, a.get("page", 1), 8)
        meta["copies"] = [self._copy_row(k) for k in rows]
        return meta

    def _t_get_copy(self, a):
        self._get("copies", a["copy_id"], "copy")
        row = self._copy_row(a["copy_id"])
        c = self.state["copies"][a["copy_id"]]
        row["borrower"] = c["borrower"]
        row["format"] = self.state["titles"][c["title"]]["format"]
        return row

    def _t_list_holds(self, a):
        if "title_id" in a:
            self._get("titles", a["title_id"], "title")
        if "patron_id" in a:
            self._get("patrons", a["patron_id"], "patron")
        if "pickup_branch" in a:
            self._branch(a["pickup_branch"])
        rows = [k for _, k in sorted((h["placed"], k) for k, h in self.state["holds"].items()
                                     if ("title_id" not in a or h["title"] == a["title_id"])
                                     and ("patron_id" not in a or h["patron"] == a["patron_id"])
                                     and ("pickup_branch" not in a or h["pickup"] == a["pickup_branch"])
                                     and ("status" not in a or h["status"] == a["status"]))]
        meta, rows = _paged(rows, a.get("page", 1), 6)
        meta["holds"] = [self._hold_row(k) for k in rows]
        return meta

    def _t_place_hold(self, a):
        pid, tid = a["patron_id"], a["title_id"]
        p = self._get("patrons", pid, "patron")
        self._get("titles", tid, "title")
        self._branch(a["pickup_branch"], True)
        if p["blocked"]:
            raise _Rule("patron %s is blocked (%s) and cannot place holds" % (pid, p["block_reason"]))
        for k in self._active(pid):
            if self.state["holds"][k]["title"] == tid:
                raise _Rule("patron %s already has an active hold on %s (%s)" % (pid, tid, k))
        for k, c in self.state["copies"].items():
            if c["status"] == "on_loan" and c["borrower"] == pid and c["title"] == tid:
                raise _Rule("patron %s currently has copy %s of %s on loan" % (pid, k, tid))
        if len(self._active(pid)) >= self._limit(pid):
            raise _Rule("patron %s already has %d active holds; the limit for a %s patron is %d" % (
                pid, len(self._active(pid)), p["type"], self._limit(pid)))
        hid = self._new_id("next_hold", "HD")
        self.state["holds"][hid] = {"patron": pid, "title": tid, "pickup": a["pickup_branch"], "status": "waiting",
                                    "placed": self.state["today"], "copy": "", "ready_until": ""}
        return {"ok": True, "hold": self._hold_row(hid)}

    def _t_cancel_hold(self, a):
        h = self._get("holds", a["hold_id"], "hold")
        if h["status"] == "in_transit":
            raise _Rule("hold %s is in transit with copy %s; receive the copy before cancelling" % (
                a["hold_id"], h["copy"]))
        if h["status"] not in ("waiting", "suspended", "ready"):
            raise _Rule("hold %s is already %s" % (a["hold_id"], h["status"]))
        if h["status"] == "ready":
            c = self.state["copies"][h["copy"]]
            c["status"], c["hold"] = "on_shelf", ""
        h["status"], h["copy"], h["ready_until"] = "cancelled", "", ""
        return {"ok": True, "hold_id": a["hold_id"], "status": "cancelled"}

    def _t_fill_hold(self, a):
        hid, cid = a["hold_id"], a["copy_id"]
        h = self._get("holds", hid, "hold")
        c = self._get("copies", cid, "copy")
        if h["status"] != "waiting":
            raise _Rule("hold %s is %s, not waiting" % (hid, h["status"]))
        if c["title"] != h["title"]:
            raise _Rule("copy %s is a copy of %s, but hold %s is for %s" % (cid, c["title"], hid, h["title"]))
        if self.state["patrons"][h["patron"]]["blocked"]:
            raise _Rule("patron %s is blocked; their holds are passed over" % h["patron"])
        self._branch(h["pickup"], True)
        if c["status"] != "on_shelf":
            raise _Rule("copy %s is %s, not on the open shelf" % (cid, c["status"]))
        if c["condition"] == "damaged":
            raise _Rule("copy %s is damaged and cannot fill holds" % cid)
        b = self.state["branches"][c["location"]]
        if b["status"] != "open":
            raise _Rule("copy %s is at %s, which is closed; it cannot be sent anywhere" % (cid, c["location"]))
        for _, k in self._queue(h["title"]):
            if self._eligible(self.state["holds"][k]):
                if k != hid:
                    raise _Rule("queue order: %s (placed %s) is the oldest eligible waiting hold on %s and must be "
                                "filled first" % (k, self.state["holds"][k]["placed"], h["title"]))
                break
        h["copy"], c["hold"] = cid, hid
        if c["location"] == h["pickup"]:
            h["status"], h["ready_until"] = "ready", _plus_days(self.state["today"], self.state["policy"]["hold_shelf_days"])
            c["status"] = "on_hold_shelf"
        else:
            h["status"], c["status"], c["destination"] = "in_transit", "in_transit", h["pickup"]
        return {"ok": True, "hold": self._hold_row(hid), "copy": self._copy_row(cid)}

    def _t_receive_copy(self, a):
        cid = a["copy_id"]
        c = self._get("copies", cid, "copy")
        if c["status"] != "in_transit":
            raise _Rule("copy %s is not in transit (status %s)" % (cid, c["status"]))
        c["location"], c["destination"] = c["destination"], ""
        if c["hold"]:
            h = self.state["holds"][c["hold"]]
            h["status"] = "ready"
            h["ready_until"] = _plus_days(self.state["today"], self.state["policy"]["hold_shelf_days"])
            c["status"] = "on_hold_shelf"
        else:
            c["status"] = "on_shelf"
        return {"ok": True, "copy": self._copy_row(cid)}

    def _t_update_hold(self, a):
        h = self._get("holds", a["hold_id"], "hold")
        if h["status"] == "suspended":
            raise _Rule("hold %s is suspended; reactivate it before changing the pickup branch" % a["hold_id"])
        if h["status"] != "waiting":
            raise _Rule("hold %s is %s; only waiting holds can be redirected" % (a["hold_id"], h["status"]))
        self._branch(a["pickup_branch"], True)
        if h["pickup"] == a["pickup_branch"]:
            raise _Rule("hold %s already has pickup branch %s" % (a["hold_id"], h["pickup"]))
        h["pickup"] = a["pickup_branch"]
        return {"ok": True, "hold": self._hold_row(a["hold_id"])}

    def _t_set_hold_suspended(self, a):
        h = self._get("holds", a["hold_id"], "hold")
        if a["suspended"]:
            if h["status"] != "waiting":
                raise _Rule("only waiting holds can be suspended; %s is %s" % (a["hold_id"], h["status"]))
            h["status"] = "suspended"
        else:
            if h["status"] != "suspended":
                raise _Rule("hold %s is %s, not suspended" % (a["hold_id"], h["status"]))
            h["status"] = "waiting"
        return {"ok": True, "hold": self._hold_row(a["hold_id"])}

    def _t_set_copy_status(self, a):
        c = self._get("copies", a["copy_id"], "copy")
        new = a["status"]
        if new == "on_shelf":
            if c["status"] != "missing":
                raise _Rule("only a missing copy can be put back on the shelf; %s is %s" % (a["copy_id"], c["status"]))
        elif new == "missing":
            if c["status"] != "on_shelf":
                raise _Rule("only copies on the open shelf can be marked missing; %s is %s" % (a["copy_id"], c["status"]))
        else:
            if c["status"] not in ("on_shelf", "missing"):
                raise _Rule("copy %s is %s and cannot be withdrawn now" % (a["copy_id"], c["status"]))
        c["status"] = new
        return {"ok": True, "copy": self._copy_row(a["copy_id"])}


# --------------------------------------------------------------------------
# check helpers + task registry
# --------------------------------------------------------------------------
def _fits(rec, want):
    for k, v in want.items():
        if callable(v):
            if not v(rec.get(k)):
                return False
        elif rec.get(k) != v:
            return False
    return True


def _match(fresh, wants):
    if len(fresh) != len(wants):
        return False
    for perm in itertools.permutations(range(len(fresh))):
        if all(_fits(fresh[perm[n]], wants[n]) for n in range(len(wants))):
            return True
    return False


def _is_collection(v):
    return isinstance(v, dict) and len(v) > 0 and all(isinstance(x, dict) for x in v.values())


def _final(s, i, changes=None, new=None):
    """True iff state s equals init i except for exactly the described differences.

    changes: {collection: {record_id: {field: value | predicate}}}; for non-collection top-level keys
             {key: value | predicate}.
    new:     {collection: [ {field: value | predicate}, ... ]} records that must have been created
             (matched by content, generated ids ignored). Collections not listed must have no new records.
    Counters (next_*) and the free-form activity "log" are ignored. Every record and field not mentioned
    must be identical to init.
    """
    changes = changes or {}
    new = new or {}
    if not isinstance(s, dict) or set(s.keys()) != set(i.keys()):
        return False
    for key in i:
        if key.startswith("next_") or key == "log":
            continue
        if _is_collection(i[key]):
            if not isinstance(s[key], dict):
                return False
            for rid, rec in i[key].items():
                got = s[key].get(rid)
                if not isinstance(got, dict) or set(got.keys()) != set(rec.keys()):
                    return False
                ch = changes.get(key, {}).get(rid, {})
                for f in rec:
                    want = ch[f] if f in ch else rec[f]
                    if callable(want):
                        if not want(got[f]):
                            return False
                    elif got[f] != want:
                        return False
            fresh = [s[key][k] for k in sorted(s[key]) if k not in i[key]]
            if not _match(fresh, new.get(key, [])):
                return False
        elif key in changes:
            want = changes[key]
            if callable(want):
                if not want(s[key]):
                    return False
            elif s[key] != want:
                return False
        elif s[key] != i[key]:
            return False
    return True


def _any(*options):
    """Pass if any of several complete (changes, new) outcomes holds."""
    return lambda s, i: any(_final(s, i, changes=c, new=n) for c, n in options)


def _set(xs):
    xs = sorted(xs)
    return lambda v: isinstance(v, list) and sorted(v) == xs


TASKS = []


def _add(tid, env, init, prompt, reference, trap, check, ref_errors=(), mentions=(), creates=(), seq=()):
    """mentions: strings the prompt names that must exist in init (checked mechanically, with the ids/quoted
    names the verifier extracts itself). creates: prompt strings that are to be created, exempt from that check.
    seq: groups of reference indices whose relative order is part of the task (not permuted by the order test)."""
    frozen = copy.deepcopy(init)
    TASKS.append({
        "id": tid, "env": env, "init": init, "prompt": prompt,
        "reference": reference, "ref_errors": list(ref_errors), "trap": trap,
        "mentions": list(mentions), "creates": list(creates), "seq": [list(g) for g in seq],
        "check": (lambda state, _c=check, _i=frozen: bool(_c(state, _i))),
    })


# --------------------------------------------------------------------------
# library fixture + tasks T01-T08
# --------------------------------------------------------------------------
def _lib():
    titles = {}
    for tid, title, author, fmt, year in [
        ("BK-110", "The Salt Road", "Mara Quist", "book", 2025), ("BK-111", "The Salt Road", "Mara Quist", "large_print", 2025),
        ("BK-112", "The Salt Roads", "Ellis Bray", "book", 2019), ("BK-113", "The Salt Road", "Mara Quist", "audiobook", 2025),
        ("BK-120", "Moonfall Harbour", "Tove Arnesen", "book", 2024), ("BK-121", "Moonfall", "Jun Takeda", "book", 2026),
        ("BK-130", "Ashes of Ilmar", "Senna Holt", "book", 2026), ("BK-131", "Ashes of Ilmar", "Senna Holt", "large_print", 2026),
        ("BK-140", "A Field Guide to Fungi", "Rhoda Penn", "book", 2021), ("BK-150", "Glass Orchard", "Ivo Marek", "book", 2025),
        ("BK-160", "The Night Ferry", "Lydia Crane", "book", 2023), ("BK-161", "Night Ferry", "Lydia Crane", "dvd", 2024),
        ("BK-170", "Small Engines", "Pete Osgood", "book", 2018), ("BK-180", "Winter Count", "Ama Boateng", "book", 2026),
        ("BK-190", "Paper Lanterns", "Keiko Arai", "book", 2025),
    ]:
        titles[tid] = {"title": title, "author": author, "format": fmt, "year": year}
    patrons = {}
    for pid, name, typ, home, issued, blocked in [
        ("P-201", "Ines Varga", "adult", "NOR", "2015-04-11", ""), ("P-202", "Ines Vargas", "adult", "CEN", "2021-09-02", ""),
        ("P-203", "Tomasz Wilk", "adult", "RIV", "2018-01-20", ""), ("P-204", "Amara Osei", "adult", "RIV", "2020-06-14", "lost items"),
        ("P-205", "Mia Okonkwo", "child", "CEN", "2023-02-03", ""), ("P-206", "Mia Okonkwo-Reed", "adult", "NOR", "2016-10-30", ""),
        ("P-207", "Felix Brandauer", "adult", "HIL", "2012-05-05", ""), ("P-208", "Grace Liang", "adult", "CEN", "2019-03-17", ""),
        ("P-209", "Yusuf Demir", "adult", "NOR", "2017-08-08", ""), ("P-210", "Lena Fischbach", "adult", "HIL", "2022-11-21", ""),
        ("P-211", "Owen Price", "staff", "CEN", "2014-01-06", ""), ("P-212", "Priya Nair", "adult", "RIV", "2020-02-29", ""),
        ("P-213", "Hannah Cole", "adult", "HIL", "2024-07-01", ""), ("P-214", "Jonah Reyes", "child", "RIV", "2024-09-09", ""),
        ("P-215", "Sofia Marin", "adult", "CEN", "2013-12-12", ""), ("P-217", "Daniel Kwan", "adult", "NOR", "2019-03-02", ""),
        ("P-218", "Daniel Kwan", "adult", "NOR", "2026-11-01", ""), ("P-219", "Rosa Lindt", "adult", "NOR", "2011-07-15", ""),
        ("P-220", "Kai Moreno", "adult", "CEN", "2021-05-19", "unreturned items"), ("P-221", "Danielle Kwan", "adult", "RIV", "2022-03-03", ""),
    ]:
        patrons[pid] = {"name": name, "type": typ, "home": home, "card_issued": issued, "blocked": bool(blocked),
                        "block_reason": blocked}
    copies = {}
    for cid, tid, loc, status, cond, dest, hold, borrower, due in [
        ("CP-1101", "BK-110", "RIV", "on_shelf", "good", "", "", "", ""),
        ("CP-1102", "BK-110", "RIV", "on_shelf", "damaged", "", "", "", ""),
        ("CP-1103", "BK-110", "CEN", "on_shelf", "good", "", "", "", ""),
        ("CP-1104", "BK-110", "NOR", "on_loan", "good", "", "", "P-209", "2026-11-10"),
        ("CP-1105", "BK-110", "HIL", "on_shelf", "good", "", "", "", ""),
        ("CP-1106", "BK-110", "CEN", "withdrawn", "worn", "", "", "", ""),
        ("CP-1107", "BK-110", "RIV", "on_shelf", "worn", "", "", "", ""),
        ("CP-1108", "BK-110", "NOR", "missing", "good", "", "", "", ""),
        ("CP-1111", "BK-111", "NOR", "on_shelf", "good", "", "", "", ""),
        ("CP-1112", "BK-111", "CEN", "on_loan", "good", "", "", "P-206", "2026-11-19"),
        ("CP-1121", "BK-112", "NOR", "on_hold_shelf", "good", "", "HD-596", "", ""),
        ("CP-1122", "BK-112", "CEN", "on_loan", "worn", "", "", "P-215", "2026-11-07"),
        ("CP-1131", "BK-113", "CEN", "on_shelf", "good", "", "", "", ""),
        ("CP-1201", "BK-120", "CEN", "on_hold_shelf", "good", "", "HD-591", "", ""),
        ("CP-1202", "BK-120", "RIV", "on_loan", "good", "", "", "P-201", "2026-11-14"),
        ("CP-1203", "BK-120", "CEN", "in_transit", "damaged", "RIV", "", "", ""),
        ("CP-1211", "BK-121", "NOR", "on_shelf", "good", "", "", "", ""),
        ("CP-1301", "BK-130", "CEN", "in_transit", "good", "RIV", "HD-540", "", ""),
        ("CP-1302", "BK-130", "NOR", "on_hold_shelf", "good", "", "HD-552", "", ""),
        ("CP-1303", "BK-130", "RIV", "on_loan", "good", "", "", "P-201", "2026-11-12"),
        ("CP-1304", "BK-130", "CEN", "on_loan", "good", "", "", "P-208", "2026-11-21"),
        ("CP-1311", "BK-131", "RIV", "on_shelf", "good", "", "", "", ""),
        ("CP-1401", "BK-140", "CEN", "on_hold_shelf", "worn", "", "HD-590", "", ""),
        ("CP-1402", "BK-140", "HIL", "on_hold_shelf", "good", "", "HD-523", "", ""),
        ("CP-1403", "BK-140", "RIV", "on_shelf", "good", "", "", "", ""),
        ("CP-1404", "BK-140", "NOR", "missing", "damaged", "", "", "", ""),
        ("CP-1501", "BK-150", "RIV", "in_transit", "good", "NOR", "", "", ""),
        ("CP-1502", "BK-150", "CEN", "on_loan", "good", "", "", "P-203", "2026-11-09"),
        ("CP-1601", "BK-160", "CEN", "on_hold_shelf", "good", "", "HD-536", "", ""),
        ("CP-1602", "BK-160", "RIV", "on_loan", "worn", "", "", "P-212", "2026-11-16"),
        ("CP-1611", "BK-161", "NOR", "in_transit", "good", "RIV", "", "", ""),
        ("CP-1612", "BK-161", "CEN", "on_hold_shelf", "good", "", "HD-597", "", ""),
        ("CP-1701", "BK-170", "CEN", "in_transit", "good", "RIV", "", "", ""),
        ("CP-1702", "BK-170", "NOR", "on_hold_shelf", "good", "", "HD-565", "", ""),
        ("CP-1801", "BK-180", "CEN", "on_hold_shelf", "good", "", "HD-594", "", ""),
        ("CP-1901", "BK-190", "CEN", "on_hold_shelf", "good", "", "HD-595", "", ""),
        ("CP-1902", "BK-190", "RIV", "in_transit", "good", "CEN", "HD-535", "", ""),
        ("CP-1903", "BK-190", "RIV", "on_hold_shelf", "good", "", "HD-581", "", ""),
    ]:
        copies[cid] = {"title": tid, "location": loc, "status": status, "condition": cond, "destination": dest,
                       "hold": hold, "borrower": borrower, "due": due}
    holds = {}
    for hid, pid, tid, pickup, status, placed, cp, until in [
        ("HD-501", "P-203", "BK-110", "NOR", "waiting", "10-02", "", ""),
        ("HD-502", "P-208", "BK-110", "CEN", "suspended", "10-05", "", ""),
        ("HD-503", "P-204", "BK-110", "RIV", "waiting", "10-07", "", ""),
        ("HD-504", "P-212", "BK-110", "RIV", "waiting", "10-11", "", ""),
        ("HD-505", "P-210", "BK-110", "HIL", "waiting", "10-14", "", ""),
        ("HD-506", "P-215", "BK-110", "CEN", "waiting", "10-20", "", ""),
        ("HD-507", "P-211", "BK-110", "NOR", "waiting", "10-28", "", ""),
        ("HD-520", "P-207", "BK-150", "HIL", "suspended", "09-25", "", ""),
        ("HD-521", "P-209", "BK-130", "HIL", "waiting", "10-16", "", ""),
        ("HD-523", "P-213", "BK-140", "HIL", "ready", "09-01", "CP-1402", "2026-11-06"),
        ("HD-524", "P-212", "BK-160", "HIL", "fulfilled", "08-20", "", ""),
        ("HD-525", "P-208", "BK-190", "HIL", "suspended", "09-29", "", ""),
        ("HD-526", "P-219", "BK-120", "HIL", "cancelled", "09-12", "", ""),
        ("HD-530", "P-205", "BK-150", "CEN", "waiting", "09-20", "", ""),
        ("HD-531", "P-205", "BK-170", "CEN", "suspended", "09-28", "", ""),
        ("HD-532", "P-205", "BK-180", "CEN", "waiting", "10-01", "", ""),
        ("HD-533", "P-205", "BK-140", "CEN", "waiting", "10-12", "", ""),
        ("HD-534", "P-205", "BK-112", "CEN", "waiting", "10-19", "", ""),
        ("HD-535", "P-205", "BK-190", "CEN", "in_transit", "10-25", "CP-1902", ""),
        ("HD-536", "P-205", "BK-160", "CEN", "ready", "10-30", "CP-1601", "2026-11-06"),
        ("HD-537", "P-205", "BK-110", "CEN", "cancelled", "10-31", "", ""),
        ("HD-540", "P-212", "BK-130", "RIV", "in_transit", "10-03", "CP-1301", ""),
        ("HD-541", "P-220", "BK-170", "CEN", "waiting", "10-02", "", ""),
        ("HD-542", "P-206", "BK-170", "NOR", "waiting", "10-14", "", ""),
        ("HD-550", "P-217", "BK-110", "NOR", "waiting", "10-29", "", ""),
        ("HD-551", "P-217", "BK-150", "CEN", "waiting", "09-15", "", ""),
        ("HD-552", "P-217", "BK-130", "NOR", "ready", "10-27", "CP-1302", "2026-11-08"),
        ("HD-553", "P-217", "BK-160", "RIV", "suspended", "10-03", "", ""),
        ("HD-554", "P-217", "BK-190", "NOR", "waiting", "10-08", "", ""),
        ("HD-555", "P-217", "BK-180", "CEN", "cancelled", "09-01", "", ""),
        ("HD-556", "P-218", "BK-190", "NOR", "waiting", "11-02", "", ""),
        ("HD-560", "P-219", "BK-140", "NOR", "waiting", "10-15", "", ""),
        ("HD-561", "P-219", "BK-150", "NOR", "waiting", "10-05", "", ""),
        ("HD-562", "P-219", "BK-112", "NOR", "waiting", "10-09", "", ""),
        ("HD-563", "P-219", "BK-180", "NOR", "waiting", "10-20", "", ""),
        ("HD-565", "P-219", "BK-170", "NOR", "ready", "09-10", "CP-1702", "2026-11-07"),
        ("HD-570", "P-201", "BK-150", "RIV", "waiting", "09-30", "", ""),
        ("HD-571", "P-206", "BK-180", "NOR", "suspended", "10-03", "", ""),
        ("HD-580", "P-214", "BK-160", "RIV", "waiting", "10-01", "", ""),
        ("HD-581", "P-214", "BK-190", "RIV", "ready", "09-22", "CP-1903", "2026-11-09"),
        ("HD-582", "P-214", "BK-170", "RIV", "waiting", "10-22", "", ""),
        ("HD-583", "P-214", "BK-120", "RIV", "suspended", "10-26", "", ""),
        ("HD-590", "P-215", "BK-140", "CEN", "ready", "09-05", "CP-1401", "2026-11-02"),
        ("HD-591", "P-211", "BK-120", "CEN", "ready", "09-20", "CP-1201", "2026-11-03"),
        ("HD-592", "P-220", "BK-120", "CEN", "waiting", "10-06", "", ""),
        ("HD-593", "P-202", "BK-120", "NOR", "waiting", "10-09", "", ""),
        ("HD-594", "P-208", "BK-180", "CEN", "ready", "09-14", "CP-1801", "2026-11-04"),
        ("HD-595", "P-201", "BK-190", "CEN", "ready", "09-18", "CP-1901", "2026-10-31"),
        ("HD-596", "P-206", "BK-112", "NOR", "ready", "09-25", "CP-1121", "2026-11-01"),
        ("HD-597", "P-209", "BK-161", "CEN", "ready", "10-12", "CP-1612", "2026-11-03"),
    ]:
        holds[hid] = {"patron": pid, "title": tid, "pickup": pickup, "status": status, "placed": "2026-" + placed,
                      "copy": cp, "ready_until": until}
    st = {"today": "2026-11-04",
          "policy": {"hold_shelf_days": 7, "max_active_holds": {"adult": 6, "child": 4, "staff": 10}},
          "branches": {"CEN": {"name": "Central", "status": "open", "reopens": "-"},
                       "NOR": {"name": "Northgate", "status": "open", "reopens": "-"},
                       "RIV": {"name": "Riverside", "status": "open", "reopens": "-"},
                       "HIL": {"name": "Hillcrest", "status": "closed", "reopens": "2027-02-01"}},
          "titles": titles, "patrons": patrons, "copies": copies, "holds": holds, "next_hold": 600}
    # sanity: the fixture is internally consistent
    for k, c in copies.items():
        if c["hold"]:
            h = holds[c["hold"]]
            assert h["copy"] == k and h["title"] == c["title"], k
            assert (c["status"], h["status"]) in (("on_hold_shelf", "ready"), ("in_transit", "in_transit")), k
            assert c["status"] != "on_hold_shelf" or c["location"] == h["pickup"], k
            assert c["status"] != "in_transit" or c["destination"] == h["pickup"], k
        assert (c["status"] == "on_loan") == bool(c["borrower"]), k
    for k, h in holds.items():
        assert (h["status"] in ("ready", "in_transit")) == bool(h["copy"]), k
    return st


_RDY = "2026-11-11"   # today + hold_shelf_days


def _c01(s, i):
    opts = []
    for x, y in (("CP-1101", "CP-1107"), ("CP-1107", "CP-1101")):
        opts.append(({
            "holds": {"HD-501": {"status": "in_transit", "copy": x},
                      "HD-504": {"status": "ready", "copy": y, "ready_until": _RDY},
                      "HD-506": {"status": "ready", "copy": "CP-1103", "ready_until": _RDY}},
            "copies": {x: {"status": "in_transit", "destination": "NOR", "hold": "HD-501"},
                       y: {"status": "on_hold_shelf", "hold": "HD-504"},
                       "CP-1103": {"status": "on_hold_shelf", "hold": "HD-506"}}}, None))
    return _any(*opts)(s, i)


_add("T01", "library", _lib(),
     "Copies of 'The Salt Road' are sitting on our shelves while people wait for it. Hand out every copy of the "
     "regular book edition that can be used right now to the waiting holds, following the library's queue rules "
     "(the large print and audiobook editions have their own queues, and 'The Salt Roads' is a different book). "
     "The queue decides which holds get a copy; where there is a choice of which copy goes to which of those holds, "
     "arrange it so that as few copies as possible have to travel between branches. Copies that can't be used stay "
     "where they are, and nothing else should change.",
     [["get_library_info", {}], ["search_titles", {"query": "Salt Road"}],
      ["list_copies", {"title_id": "BK-110"}], ["list_holds", {"title_id": "BK-110"}],
      ["list_holds", {"title_id": "BK-110", "page": 2}],
      ["fill_hold", {"hold_id": "HD-501", "copy_id": "CP-1101"}],
      ["fill_hold", {"hold_id": "HD-503", "copy_id": "CP-1107"}],
      ["fill_hold", {"hold_id": "HD-504", "copy_id": "CP-1107"}],
      ["fill_hold", {"hold_id": "HD-506", "copy_id": "CP-1103"}]],
     [["fill_hold", {"hold_id": "HD-501", "copy_id": "CP-1103"}],
      ["fill_hold", {"hold_id": "HD-504", "copy_id": "CP-1101"}],
      ["fill_hold", {"hold_id": "HD-506", "copy_id": "CP-1107"}]],
     _c01, ref_errors=[6], mentions=["The Salt Road", "The Salt Roads"])

_add("T02", "library", _lib(),
     "Hillcrest has closed for refurbishment, so please re-route the holds that were going to be collected there. "
     "Every Hillcrest hold that is still waiting or suspended should be collected at the patron's home branch "
     "instead, or at Central if the patron's home branch is Hillcrest itself. Suspended holds must still be "
     "suspended when you're done - those patrons asked for a pause and that shouldn't change. Anything already "
     "sitting on the Hillcrest hold shelf stays where it is (staff will phone those patrons), old fulfilled or "
     "cancelled holds are history and stay as they are, and holds for other branches are not part of this.",
     [["get_library_info", {}], ["list_holds", {"pickup_branch": "HIL"}],
      ["list_holds", {"pickup_branch": "HIL", "page": 2}],
      ["get_patron", {"patron_id": "P-210"}], ["get_patron", {"patron_id": "P-207"}],
      ["get_patron", {"patron_id": "P-209"}], ["get_patron", {"patron_id": "P-208"}],
      ["update_hold", {"hold_id": "HD-505", "pickup_branch": "CEN"}],
      ["update_hold", {"hold_id": "HD-521", "pickup_branch": "NOR"}],
      ["set_hold_suspended", {"hold_id": "HD-520", "suspended": False}],
      ["update_hold", {"hold_id": "HD-520", "pickup_branch": "CEN"}],
      ["set_hold_suspended", {"hold_id": "HD-520", "suspended": True}],
      ["set_hold_suspended", {"hold_id": "HD-525", "suspended": False}],
      ["update_hold", {"hold_id": "HD-525", "pickup_branch": "CEN"}],
      ["set_hold_suspended", {"hold_id": "HD-525", "suspended": True}]],
     [["update_hold", {"hold_id": "HD-505", "pickup_branch": "CEN"}],
      ["update_hold", {"hold_id": "HD-521", "pickup_branch": "NOR"}],
      ["set_hold_suspended", {"hold_id": "HD-520", "suspended": False}],
      ["update_hold", {"hold_id": "HD-520", "pickup_branch": "CEN"}],
      ["set_hold_suspended", {"hold_id": "HD-525", "suspended": False}],
      ["update_hold", {"hold_id": "HD-525", "pickup_branch": "CEN"}]],
     lambda s, i: _final(s, i, changes={"holds": {
         "HD-505": {"pickup": "CEN"}, "HD-521": {"pickup": "NOR"}, "HD-520": {"pickup": "CEN"},
         "HD-525": {"pickup": "CEN"}}}),
     mentions=["Hillcrest", "Central"])

_add("T03", "library", _lib(),
     "Children's cards are now limited to 4 active holds, and Mia Okonkwo (the child - not Mia Okonkwo-Reed) is "
     "over that limit. She also wants a hold on the book 'Moonfall' (not 'Moonfall Harbour'), to collect at her "
     "home branch. Make room by cancelling her holds newest first, going by the date each was placed, but never "
     "cancel a hold whose copy is already ready for her or already on its way to her: skip those and take the next "
     "newest instead. Cancel only as many as needed for her to end up exactly at the limit once the new hold is "
     "placed, then place it. Nothing else should change.",
     [["find_patrons", {"query": "Mia Okonkwo"}], ["get_patron", {"patron_id": "P-205"}],
      ["list_holds", {"patron_id": "P-205"}], ["list_holds", {"patron_id": "P-205", "page": 2}],
      ["search_titles", {"query": "Moonfall"}],
      ["cancel_hold", {"hold_id": "HD-534"}], ["cancel_hold", {"hold_id": "HD-533"}],
      ["cancel_hold", {"hold_id": "HD-532"}], ["cancel_hold", {"hold_id": "HD-531"}],
      ["place_hold", {"patron_id": "P-205", "title_id": "BK-121", "pickup_branch": "CEN"}]],
     [["cancel_hold", {"hold_id": "HD-536"}], ["cancel_hold", {"hold_id": "HD-534"}],
      ["cancel_hold", {"hold_id": "HD-533"}],
      ["place_hold", {"patron_id": "P-205", "title_id": "BK-121", "pickup_branch": "CEN"}]],
     lambda s, i: _final(s, i, changes={"holds": dict(
         (h, {"status": "cancelled"}) for h in ("HD-534", "HD-533", "HD-532", "HD-531"))},
         new={"holds": [{"patron": "P-205", "title": "BK-121", "pickup": "CEN", "status": "waiting",
                         "placed": "2026-11-04", "copy": "", "ready_until": ""}]}),
     mentions=["Mia Okonkwo", "Mia Okonkwo-Reed"], seq=[[5, 9], [6, 9], [7, 9], [8, 9]])

_add("T04", "library", _lib(),
     "The Riverside delivery van just came in. The driver's sheet lists CP-1301, CP-1701, CP-1501, CP-1403, "
     "CP-1203 and CP-1611. Check in everything on that sheet that was actually addressed to Riverside and has not "
     "been checked in yet; a copy addressed to another branch stays on the van, in transit, and a copy that was "
     "already checked in needs nothing. A copy that arrived damaged must not be lent out: withdraw it once it has "
     "been checked in. Every other copy that arrives without a hold attached should go straight to the next hold "
     "in line for its title, if there is one, following the queue rules. Change nothing else.",
     [["get_library_info", {}], ["list_copies", {"status": "in_transit"}], ["get_copy", {"copy_id": "CP-1403"}],
      ["receive_copy", {"copy_id": "CP-1301"}], ["receive_copy", {"copy_id": "CP-1611"}],
      ["receive_copy", {"copy_id": "CP-1701"}], ["receive_copy", {"copy_id": "CP-1203"}],
      ["set_copy_status", {"copy_id": "CP-1203", "status": "withdrawn"}],
      ["list_holds", {"title_id": "BK-170"}], ["list_holds", {"title_id": "BK-161"}],
      ["fill_hold", {"hold_id": "HD-541", "copy_id": "CP-1701"}],
      ["fill_hold", {"hold_id": "HD-542", "copy_id": "CP-1701"}]],
     [["receive_copy", {"copy_id": "CP-1301"}], ["receive_copy", {"copy_id": "CP-1611"}],
      ["receive_copy", {"copy_id": "CP-1701"}], ["receive_copy", {"copy_id": "CP-1203"}],
      ["receive_copy", {"copy_id": "CP-1501"}], ["set_copy_status", {"copy_id": "CP-1203", "status": "withdrawn"}],
      ["fill_hold", {"hold_id": "HD-542", "copy_id": "CP-1701"}]],
     lambda s, i: _final(s, i, changes={
         "copies": {"CP-1301": {"location": "RIV", "destination": "", "status": "on_hold_shelf"},
                    "CP-1611": {"location": "RIV", "destination": "", "status": "on_shelf"},
                    "CP-1701": {"location": "RIV", "destination": "NOR", "status": "in_transit", "hold": "HD-542"},
                    "CP-1203": {"location": "RIV", "destination": "", "status": "withdrawn"}},
         "holds": {"HD-540": {"status": "ready", "ready_until": _RDY},
                   "HD-542": {"status": "in_transit", "copy": "CP-1701"}}}),
     ref_errors=[10], mentions=["Riverside"])

_add("T05", "library", _lib(),
     "Daniel Kwan was issued a new library card this month and his old card is being retired. Move his holds "
     "over to the new card: for every hold on the old card that is waiting or suspended, give the new card a hold "
     "on the same title with the same pickup branch (one that was suspended should be suspended on the new card "
     "too), and cancel the old one. If the new card already has a hold on that title, simply cancel the old one. "
     "The hold that is already ready for him stays on the old card - he will collect it with that card. Careful: "
     "Danielle Kwan is a different person. Nothing else should change.",
     [["find_patrons", {"query": "Daniel Kwan"}], ["list_holds", {"patron_id": "P-217"}],
      ["list_holds", {"patron_id": "P-218"}],
      ["place_hold", {"patron_id": "P-218", "title_id": "BK-110", "pickup_branch": "NOR"}],
      ["place_hold", {"patron_id": "P-218", "title_id": "BK-150", "pickup_branch": "CEN"}],
      ["place_hold", {"patron_id": "P-218", "title_id": "BK-160", "pickup_branch": "RIV"}],
      ["set_hold_suspended", {"hold_id": "HD-602", "suspended": True}],
      ["cancel_hold", {"hold_id": "HD-550"}], ["cancel_hold", {"hold_id": "HD-551"}],
      ["cancel_hold", {"hold_id": "HD-553"}], ["cancel_hold", {"hold_id": "HD-554"}]],
     [["place_hold", {"patron_id": "P-218", "title_id": "BK-110", "pickup_branch": "NOR"}],
      ["place_hold", {"patron_id": "P-218", "title_id": "BK-150", "pickup_branch": "CEN"}],
      ["place_hold", {"patron_id": "P-218", "title_id": "BK-160", "pickup_branch": "RIV"}],
      ["cancel_hold", {"hold_id": "HD-550"}], ["cancel_hold", {"hold_id": "HD-551"}],
      ["cancel_hold", {"hold_id": "HD-553"}], ["cancel_hold", {"hold_id": "HD-554"}]],
     lambda s, i: _final(s, i, changes={"holds": dict(
         (h, {"status": "cancelled"}) for h in ("HD-550", "HD-551", "HD-553", "HD-554"))},
         new={"holds": [
             {"patron": "P-218", "title": "BK-110", "pickup": "NOR", "status": "waiting", "placed": "2026-11-04",
              "copy": "", "ready_until": ""},
             {"patron": "P-218", "title": "BK-150", "pickup": "CEN", "status": "waiting", "placed": "2026-11-04",
              "copy": "", "ready_until": ""},
             {"patron": "P-218", "title": "BK-160", "pickup": "RIV", "status": "suspended", "placed": "2026-11-04",
              "copy": "", "ready_until": ""}]}),
     mentions=["Daniel Kwan", "Danielle Kwan"], seq=[[3, 4, 5, 6]])

_add("T06", "library", _lib(),
     "Rosa Lindt is working in another city for a few weeks. Suspend each of her waiting holds on which she is "
     "first or second in line, so that no copy gets sent to her while she's away. Her place in line on a title is "
     "1 plus the number of holds on that same title that were placed before hers and are still waiting - "
     "suspended, ready, cancelled and fulfilled holds don't count. Her other waiting holds won't come up for a "
     "while; switch their pickup to Central, where she will be on Fridays. Leave the copy that is already ready for "
     "her alone, and change nothing else.",
     [["find_patrons", {"query": "Rosa Lindt"}], ["list_holds", {"patron_id": "P-219"}],
      ["list_holds", {"title_id": "BK-140"}], ["list_holds", {"title_id": "BK-150"}],
      ["list_holds", {"title_id": "BK-112"}], ["list_holds", {"title_id": "BK-180"}],
      ["set_hold_suspended", {"hold_id": "HD-560", "suspended": True}],
      ["set_hold_suspended", {"hold_id": "HD-562", "suspended": True}],
      ["set_hold_suspended", {"hold_id": "HD-563", "suspended": True}],
      ["update_hold", {"hold_id": "HD-561", "pickup_branch": "CEN"}]],
     [["set_hold_suspended", {"hold_id": "HD-560", "suspended": True}],
      ["set_hold_suspended", {"hold_id": "HD-562", "suspended": True}],
      ["update_hold", {"hold_id": "HD-561", "pickup_branch": "CEN"}],
      ["update_hold", {"hold_id": "HD-563", "pickup_branch": "CEN"}]],
     lambda s, i: _final(s, i, changes={"holds": {
         "HD-560": {"status": "suspended"}, "HD-562": {"status": "suspended"}, "HD-563": {"status": "suspended"},
         "HD-561": {"pickup": "CEN"}}}),
     mentions=["Rosa Lindt", "Central"])

_add("T07", "library", _lib(),
     "The Riverside book club reads 'Ashes of Ilmar' next month (the regular book, not the large print). Make sure "
     "each member has a hold on it for collection at Riverside: Tomasz Wilk, Amara Osei, Priya Nair, Jonah Reyes, "
     "Yusuf Demir and Ines Varga. A member who already has a hold on it for Riverside needs nothing; if a member's "
     "existing hold is for another branch, switch it to Riverside rather than placing a second one. Anyone who has "
     "the book out on loan right now doesn't need a hold. If a member is at their hold limit, cancel their newest "
     "waiting hold (not a suspended or ready one) to make room. If the rules simply don't allow a member to have a "
     "hold, leave them out. Change nothing else.",
     [["search_titles", {"query": "Ashes of Ilmar"}], ["list_holds", {"title_id": "BK-130"}],
      ["find_patrons", {"query": "Tomasz Wilk"}], ["find_patrons", {"query": "Amara Osei"}],
      ["find_patrons", {"query": "Jonah Reyes"}], ["find_patrons", {"query": "Ines Varga"}],
      ["get_patron", {"patron_id": "P-201"}], ["get_patron", {"patron_id": "P-204"}],
      ["get_patron", {"patron_id": "P-214"}], ["list_holds", {"patron_id": "P-214"}],
      ["place_hold", {"patron_id": "P-203", "title_id": "BK-130", "pickup_branch": "RIV"}],
      ["update_hold", {"hold_id": "HD-521", "pickup_branch": "RIV"}],
      ["cancel_hold", {"hold_id": "HD-582"}],
      ["place_hold", {"patron_id": "P-214", "title_id": "BK-130", "pickup_branch": "RIV"}]],
     [["place_hold", {"patron_id": "P-203", "title_id": "BK-130", "pickup_branch": "RIV"}],
      ["update_hold", {"hold_id": "HD-521", "pickup_branch": "RIV"}],
      ["cancel_hold", {"hold_id": "HD-583"}],
      ["place_hold", {"patron_id": "P-214", "title_id": "BK-130", "pickup_branch": "RIV"}],
      ["place_hold", {"patron_id": "P-202", "title_id": "BK-130", "pickup_branch": "RIV"}]],
     lambda s, i: _final(s, i, changes={"holds": {"HD-521": {"pickup": "RIV"}, "HD-582": {"status": "cancelled"}}},
                         new={"holds": [
                             {"patron": "P-203", "title": "BK-130", "pickup": "RIV", "status": "waiting",
                              "placed": "2026-11-04", "copy": "", "ready_until": ""},
                             {"patron": "P-214", "title": "BK-130", "pickup": "RIV", "status": "waiting",
                              "placed": "2026-11-04", "copy": "", "ready_until": ""}]}),
     mentions=["Riverside", "Tomasz Wilk", "Amara Osei", "Priya Nair", "Jonah Reyes", "Yusuf Demir", "Ines Varga"],
     seq=[[12, 13]])

_add("T08", "library", _lib(),
     "Please clear the expired holds off the Central hold shelf. A ready hold has expired once its last pickup day "
     "has passed; cancel every expired hold at Central (other branches' hold shelves are not part of this). Each copy "
     "freed up that way should go straight to the next hold in line for its title according to the queue rules, "
     "wherever that patron collects; if nobody is waiting for the title, the copy just stays on the shelf. Holds "
     "that are still inside their pickup window must stay exactly as they are, and nothing else should change.",
     [["get_library_info", {}], ["list_holds", {"pickup_branch": "CEN", "status": "ready"}],
      ["cancel_hold", {"hold_id": "HD-590"}], ["cancel_hold", {"hold_id": "HD-591"}],
      ["cancel_hold", {"hold_id": "HD-595"}], ["cancel_hold", {"hold_id": "HD-597"}],
      ["list_holds", {"title_id": "BK-140"}], ["fill_hold", {"hold_id": "HD-533", "copy_id": "CP-1401"}],
      ["list_holds", {"title_id": "BK-120"}], ["fill_hold", {"hold_id": "HD-593", "copy_id": "CP-1201"}],
      ["list_holds", {"title_id": "BK-190"}], ["fill_hold", {"hold_id": "HD-554", "copy_id": "CP-1901"}],
      ["list_holds", {"title_id": "BK-161"}]],
     [["cancel_hold", {"hold_id": "HD-590"}], ["cancel_hold", {"hold_id": "HD-591"}],
      ["cancel_hold", {"hold_id": "HD-594"}], ["cancel_hold", {"hold_id": "HD-595"}],
      ["cancel_hold", {"hold_id": "HD-597"}], ["fill_hold", {"hold_id": "HD-533", "copy_id": "CP-1401"}],
      ["fill_hold", {"hold_id": "HD-593", "copy_id": "CP-1201"}],
      ["fill_hold", {"hold_id": "HD-554", "copy_id": "CP-1901"}]],
     lambda s, i: _final(s, i, changes={
         "holds": {"HD-590": {"status": "cancelled", "copy": "", "ready_until": ""},
                   "HD-591": {"status": "cancelled", "copy": "", "ready_until": ""},
                   "HD-595": {"status": "cancelled", "copy": "", "ready_until": ""},
                   "HD-597": {"status": "cancelled", "copy": "", "ready_until": ""},
                   "HD-533": {"status": "ready", "copy": "CP-1401", "ready_until": _RDY},
                   "HD-593": {"status": "in_transit", "copy": "CP-1201"},
                   "HD-554": {"status": "in_transit", "copy": "CP-1901"}},
         "copies": {"CP-1401": {"status": "on_hold_shelf", "hold": "HD-533"},
                    "CP-1201": {"status": "in_transit", "destination": "NOR", "hold": "HD-593"},
                    "CP-1901": {"status": "in_transit", "destination": "NOR", "hold": "HD-554"},
                    "CP-1612": {"status": "on_shelf", "hold": ""}}}),
     mentions=["Central"])


# --------------------------------------------------------------------------
# 2. IT assets: employees, devices, software licences and seats
# --------------------------------------------------------------------------
_DEPTS = ["Design", "Engineering", "Support", "Sales", "People"]
_DEV_KINDS = ["laptop", "monitor", "phone"]


class AssetsEnv(_Env):
    TOOLS = [
        _fn("get_it_policy", "Today's date and the IT asset and licence policy that the write tools enforce."),
        _fn("find_employees", "Search employees by name (case-insensitive substring) and/or department and status.",
            {"query": _s("Text that must appear in the name."), "dept": _s("Only this department.", _DEPTS),
             "status": _s("Only this employment status.", ["starting", "active", "leaving", "left"])}),
        _fn("get_employee", "One employee: department, status, end date, devices currently assigned and active "
            "software seats.", {"employee_id": _s("Employee id, e.g. E-301.")}, ["employee_id"]),
        _fn("list_devices", "Devices, filtered; 8 per page, by device id.",
            {"kind": _s("Only this kind.", _DEV_KINDS),
             "status": _s("Only this status.", ["in_use", "in_stock", "repair", "retired"]),
             "assigned_to": _s("Only devices assigned to this employee id."), "page": _PAGE}),
        _fn("list_licences", "All software licences: product, seat count, seats in use, free seats and which "
            "departments may hold a seat."),
        _fn("list_seats", "Seats of one licence, by seat id; 8 per page. Shows holder, holder's department and "
            "status, date assigned, date last used and seat status.",
            {"licence_id": _s("Licence id, e.g. SW-11."), "status": _s("Only seats in this status.", ["active", "revoked"]),
             "page": _PAGE}, ["licence_id"]),
        _fn("assign_device", "Assign an in-stock device to an employee.",
            {"device_id": _s("Device id, e.g. D-4001."), "employee_id": _s("Employee id.")}, ["device_id", "employee_id"]),
        _fn("unassign_device", "Take a device back from its holder and put it in stock or send it to repair.",
            {"device_id": _s("Device id."), "destination": _s("Where it goes.", ["in_stock", "repair"])},
            ["device_id", "destination"]),
        _fn("set_device_status", "Move an unassigned device between stock and repair.",
            {"device_id": _s("Device id."), "status": _s("New status.", ["in_stock", "repair"])}, ["device_id", "status"]),
        _fn("retire_device", "Retire an unassigned device permanently.", {"device_id": _s("Device id.")}, ["device_id"]),
        _fn("assign_seat", "Give an employee a seat on a licence.",
            {"licence_id": _s("Licence id."), "employee_id": _s("Employee id.")}, ["licence_id", "employee_id"]),
        _fn("revoke_seat", "Take an employee's active seat on a licence back.",
            {"licence_id": _s("Licence id."), "employee_id": _s("Employee id.")}, ["licence_id", "employee_id"]),
        _fn("set_seat_count", "Change the number of seats on a licence.",
            {"licence_id": _s("Licence id."), "seats": _i("New total number of seats.", 0)}, ["licence_id", "seats"]),
        _fn("update_employee", "Change an employee's department and/or employment status.",
            {"employee_id": _s("Employee id."), "dept": _s("New department.", _DEPTS),
             "status": _s("New status.", ["active", "leaving", "left"])}, ["employee_id"]),
    ]

    def _today(self):
        return _pdate(self.state["today"])

    def _eol(self, d):
        if d["kind"] != "laptop":
            return False
        t = self._today()
        cutoff = datetime.date(t.year - self.state["policy"]["laptop_max_age_years"], t.month, t.day)
        return _pdate(d["purchased"]) < cutoff

    def _seats_of(self, lic=None, emp=None):
        return [k for k, x in sorted(self.state["seats"].items()) if x["status"] == "active"
                and (lic is None or x["licence"] == lic) and (emp is None or x["employee"] == emp)]

    def _dev_row(self, k):
        d = self.state["devices"][k]
        row = dict(device_id=k, **d)
        row["assigned_to_name"] = self.state["employees"][d["assigned_to"]]["name"] if d["assigned_to"] else ""
        return row

    def _t_get_it_policy(self, a):
        p = self.state["policy"]
        return {"today": self.state["today"], "policy": {
            "laptop_max_age_years": p["laptop_max_age_years"], "seat_inactivity_days": p["inactivity_days"],
            "max_devices_per_employee": p["max_devices"], "starter_bundles": p["bundles"],
            "rules": [
                "A laptop is end-of-life when it was purchased more than laptop_max_age_years before today (a laptop "
                "bought exactly that many years ago today is not yet end-of-life). End-of-life laptops may never be "
                "assigned to anyone, and they are the only laptops that may be retired.",
                "Only encrypted laptops may be assigned. Devices in repair or retired cannot be assigned.",
                "An employee may hold at most max_devices_per_employee devices of each kind at once.",
                "Devices and seats can only be given to employees who are starting or active.",
                "A seat is inactive when its last use was more than seat_inactivity_days days before today.",
                "Seats on a licence with allowed_depts may only be held by employees of those departments; an "
                "employee cannot be moved to a department while holding seats that department is not allowed.",
                "A licence can never have more active seats than its seat count, and the seat count cannot be set "
                "below the number of active seats.",
                "An employee can only be marked as left once they hold no devices and no active seats.",
            ]}}

    def _t_find_employees(self, a):
        if not any(k in a for k in ("query", "dept", "status")):
            raise _Rule("give at least one of query, dept, status")
        q = a.get("query", "").strip().lower()
        return {"employees": [{"employee_id": k, "name": e["name"], "dept": e["dept"], "status": e["status"],
                               "end_date": e["end_date"]}
                              for k, e in sorted(self.state["employees"].items())
                              if q in e["name"].lower() and ("dept" not in a or e["dept"] == a["dept"])
                              and ("status" not in a or e["status"] == a["status"])]}

    def _t_get_employee(self, a):
        e = self._get("employees", a["employee_id"], "employee")
        devs = [self._dev_row(k) for k, d in sorted(self.state["devices"].items())
                if d["assigned_to"] == a["employee_id"] and d["status"] == "in_use"]
        seats = [{"licence_id": self.state["seats"][k]["licence"],
                  "product": self.state["licences"][self.state["seats"][k]["licence"]]["product"],
                  "assigned": self.state["seats"][k]["assigned"], "last_used": self.state["seats"][k]["last_used"]}
                 for k in self._seats_of(emp=a["employee_id"])]
        return {"employee_id": a["employee_id"], "name": e["name"], "dept": e["dept"], "status": e["status"],
                "end_date": e["end_date"], "devices": devs, "seats": seats}

    def _t_list_devices(self, a):
        if "assigned_to" in a:
            self._get("employees", a["assigned_to"], "employee")
        rows = [k for k, d in sorted(self.state["devices"].items())
                if ("kind" not in a or d["kind"] == a["kind"]) and ("status" not in a or d["status"] == a["status"])
                and ("assigned_to" not in a or d["assigned_to"] == a["assigned_to"])]
        meta, rows = _paged(rows, a.get("page", 1), 8)
        meta["devices"] = [self._dev_row(k) for k in rows]
        return meta

    def _t_list_licences(self, a):
        out = []
        for k, l in sorted(self.state["licences"].items()):
            used = len(self._seats_of(lic=k))
            out.append({"licence_id": k, "product": l["product"], "seats": l["seats"], "in_use": used,
                        "free": l["seats"] - used, "allowed_depts": list(l["allowed_depts"]) or "all"})
        return {"licences": out}

    def _t_list_seats(self, a):
        self._get("licences", a["licence_id"], "licence")
        rows = [k for k, x in sorted(self.state["seats"].items()) if x["licence"] == a["licence_id"]
                and ("status" not in a or x["status"] == a["status"])]
        meta, rows = _paged(rows, a.get("page", 1), 8)
        out = []
        for k in rows:
            x = self.state["seats"][k]
            e = self.state["employees"][x["employee"]]
            out.append({"seat_id": k, "employee_id": x["employee"], "employee_name": e["name"], "dept": e["dept"],
                        "employee_status": e["status"], "assigned": x["assigned"], "last_used": x["last_used"] or "never",
                        "status": x["status"]})
        meta["licence_id"] = a["licence_id"]
        meta["seats"] = out
        return meta

    def _emp_ok(self, eid):
        e = self._get("employees", eid, "employee")
        if e["status"] not in ("starting", "active"):
            raise _Rule("employee %s is %s; only starting or active employees can receive devices or seats" % (
                eid, e["status"]))
        return e

    def _t_assign_device(self, a):
        d = self._get("devices", a["device_id"], "device")
        self._emp_ok(a["employee_id"])
        if d["status"] != "in_stock":
            raise _Rule("device %s is %s, not in stock" % (a["device_id"], d["status"]))
        if d["kind"] == "laptop":
            if self._eol(d):
                raise _Rule("laptop %s (purchased %s) is end-of-life and may not be assigned" % (a["device_id"], d["purchased"]))
            if not d["encrypted"]:
                raise _Rule("laptop %s is not encrypted and may not be assigned" % a["device_id"])
        held = [k for k, x in self.state["devices"].items() if x["assigned_to"] == a["employee_id"]
                and x["status"] == "in_use" and x["kind"] == d["kind"]]
        cap = self.state["policy"]["max_devices"][d["kind"]]
        if len(held) >= cap:
            raise _Rule("employee %s already has %d %s(s) (%s); the maximum is %d" % (
                a["employee_id"], len(held), d["kind"], ", ".join(sorted(held)), cap))
        d["status"], d["assigned_to"] = "in_use", a["employee_id"]
        return {"ok": True, "device": self._dev_row(a["device_id"])}

    def _t_unassign_device(self, a):
        d = self._get("devices", a["device_id"], "device")
        if d["status"] != "in_use":
            raise _Rule("device %s is not assigned to anyone (status %s)" % (a["device_id"], d["status"]))
        d["last_user"], d["assigned_to"], d["status"] = d["assigned_to"], "", a["destination"]
        return {"ok": True, "device": self._dev_row(a["device_id"])}

    def _t_set_device_status(self, a):
        d = self._get("devices", a["device_id"], "device")
        if d["status"] not in ("in_stock", "repair"):
            raise _Rule("device %s is %s; only unassigned devices in stock or repair can be moved" % (
                a["device_id"], d["status"]))
        if d["status"] == a["status"]:
            raise _Rule("device %s is already %s" % (a["device_id"], d["status"]))
        d["status"] = a["status"]
        return {"ok": True, "device": self._dev_row(a["device_id"])}

    def _t_retire_device(self, a):
        d = self._get("devices", a["device_id"], "device")
        if d["status"] not in ("in_stock", "repair"):
            raise _Rule("device %s is %s; take it back before retiring it" % (a["device_id"], d["status"]))
        if d["kind"] == "laptop" and not self._eol(d):
            raise _Rule("laptop %s (purchased %s) is not end-of-life and may not be retired" % (a["device_id"], d["purchased"]))
        d["status"] = "retired"
        return {"ok": True, "device": self._dev_row(a["device_id"])}

    def _t_assign_seat(self, a):
        l = self._get("licences", a["licence_id"], "licence")
        e = self._emp_ok(a["employee_id"])
        if l["allowed_depts"] and e["dept"] not in l["allowed_depts"]:
            raise _Rule("%s seats are restricted to %s; %s is in %s" % (
                l["product"], ", ".join(l["allowed_depts"]), a["employee_id"], e["dept"]))
        if self._seats_of(a["licence_id"], a["employee_id"]):
            raise _Rule("%s already has an active %s seat" % (a["employee_id"], l["product"]))
        used = len(self._seats_of(lic=a["licence_id"]))
        if used >= l["seats"]:
            raise _Rule("%s has no free seat (%d of %d in use)" % (l["product"], used, l["seats"]))
        sid = self._new_id("next_seat", "SE")
        self.state["seats"][sid] = {"licence": a["licence_id"], "employee": a["employee_id"],
                                    "assigned": self.state["today"], "last_used": "", "status": "active"}
        return {"ok": True, "seat_id": sid, "free_seats_left": l["seats"] - used - 1}

    def _t_revoke_seat(self, a):
        l = self._get("licences", a["licence_id"], "licence")
        self._get("employees", a["employee_id"], "employee")
        mine = self._seats_of(a["licence_id"], a["employee_id"])
        if not mine:
            raise _Rule("%s has no active %s seat" % (a["employee_id"], l["product"]))
        self.state["seats"][mine[0]]["status"] = "revoked"
        return {"ok": True, "seat_id": mine[0], "free_seats_now": l["seats"] - len(self._seats_of(lic=a["licence_id"]))}

    def _t_set_seat_count(self, a):
        l = self._get("licences", a["licence_id"], "licence")
        used = len(self._seats_of(lic=a["licence_id"]))
        if a["seats"] < used:
            raise _Rule("%s has %d active seats; the seat count cannot go below that" % (l["product"], used))
        if a["seats"] == l["seats"]:
            raise _Rule("%s already has %d seats" % (l["product"], l["seats"]))
        l["seats"] = a["seats"]
        return {"ok": True, "licence_id": a["licence_id"], "seats": l["seats"], "in_use": used}

    def _t_update_employee(self, a):
        e = self._get("employees", a["employee_id"], "employee")
        if "dept" not in a and "status" not in a:
            raise _Rule("give dept and/or status")
        if "dept" in a and a["dept"] != e["dept"]:
            for k in self._seats_of(emp=a["employee_id"]):
                l = self.state["licences"][self.state["seats"][k]["licence"]]
                if l["allowed_depts"] and a["dept"] not in l["allowed_depts"]:
                    raise _Rule("%s holds a %s seat, which %s is not allowed; revoke it first" % (
                        a["employee_id"], l["product"], a["dept"]))
            e["dept"] = a["dept"]
        if "status" in a and a["status"] != e["status"]:
            allowed = {"starting": ("active",), "active": ("leaving", "left"), "leaving": ("active", "left")}
            if a["status"] not in allowed.get(e["status"], ()):
                raise _Rule("cannot change status from %s to %s" % (e["status"], a["status"]))
            if a["status"] == "left":
                devs = [k for k, d in sorted(self.state["devices"].items())
                        if d["assigned_to"] == a["employee_id"] and d["status"] == "in_use"]
                seats = self._seats_of(emp=a["employee_id"])
                if devs or seats:
                    raise _Rule("%s still holds devices %s and %d active seat(s)" % (
                        a["employee_id"], ", ".join(devs) or "none", len(seats)))
                if not e["end_date"]:
                    e["end_date"] = self.state["today"]
            e["status"] = a["status"]
        return {"ok": True, "employee": {"employee_id": a["employee_id"], "name": e["name"], "dept": e["dept"],
                                         "status": e["status"]}}


def _assets():
    emps = {}
    for eid, name, dept, status, end in [
        ("E-301", "Lucia Bianchi", "Design", "active", ""), ("E-302", "Peter Holm", "Design", "active", ""),
        ("E-303", "Grace Mwangi", "Design", "active", ""), ("E-304", "Tariq Aziz", "Design", "active", ""),
        ("E-305", "Ana Ferreira", "Design", "active", ""), ("E-306", "Ana Ferreira-Lopes", "Sales", "active", ""),
        ("E-307", "Ivo Petrov", "Design", "left", "2026-10-02"), ("E-308", "Mei Tanaka", "Design", "active", ""),
        ("E-309", "Jonas Lind", "Design", "leaving", "2026-12-11"), ("E-310", "Olu Adeyemi", "Engineering", "active", ""),
        ("E-311", "Rachel Stone", "Engineering", "active", ""), ("E-312", "Marcus Hale", "Design", "leaving", "2026-11-16"),
        ("E-313", "Marcus Hales", "Sales", "leaving", "2026-12-18"), ("E-314", "Marcia Hale", "Support", "active", ""),
        ("E-315", "Dev Patel", "Engineering", "active", ""), ("E-316", "Sara Holt", "Engineering", "left", "2026-08-29"),
        ("E-317", "Kenji Mori", "Engineering", "active", ""), ("E-318", "Bea Nunez", "Support", "active", ""),
        ("E-319", "Carl Jensen", "Support", "active", ""), ("E-320", "Sven Aalto", "Engineering", "active", ""),
        ("E-321", "Hugo Lemaire", "Support", "active", ""), ("E-322", "Lina Haas", "Support", "active", ""),
        ("E-323", "Omar Farouk", "Support", "active", ""), ("E-324", "Pia Kovac", "Support", "active", ""),
        ("E-325", "Raj Iyer", "Sales", "active", ""), ("E-326", "Tess Moran", "Sales", "active", ""),
        ("E-327", "Femi Bello", "People", "active", ""), ("E-328", "Nadia Rahman", "People", "left", "2026-09-30"),
        ("E-330", "Leon Weiss", "Engineering", "active", ""), ("E-331", "Zoe Grant", "Support", "active", ""),
        ("E-332", "Ali Haidar", "Sales", "left", "2026-10-20"), ("E-333", "Ivy Chen", "Support", "active", ""),
        ("E-335", "Rhea Doyle", "Support", "left", "2026-10-09"),
        ("E-340", "Noemi Laine", "Design", "starting", ""), ("E-341", "Theo Brandvold", "Design", "starting", ""),
        ("E-342", "Ruth Ekdahl", "Engineering", "starting", ""),
    ]:
        emps[eid] = {"name": name, "dept": dept, "status": status, "end_date": end}
    devs = {}
    tp, mac = "ThinkPad T14", "MacBook Pro 14"
    for did, kind, model, bought, status, who, enc, last in [
        ("D-4001", "laptop", mac, "2024-03-10", "in_use", "E-301", True, ""),
        ("D-4002", "laptop", mac, "2023-05-02", "in_use", "E-302", True, ""),
        ("D-4003", "laptop", mac, "2025-01-15", "in_use", "E-303", True, ""),
        ("D-4004", "laptop", mac, "2022-12-01", "in_use", "E-304", True, ""),
        ("D-4005", "laptop", mac, "2024-08-19", "in_use", "E-305", True, ""),
        ("D-4006", "laptop", tp, "2023-09-09", "in_use", "E-306", True, ""),
        ("D-4007", "laptop", mac, "2025-06-30", "in_use", "E-308", True, ""),
        ("D-4008", "laptop", mac, "2023-02-14", "in_use", "E-309", True, ""),
        ("D-4009", "laptop", tp, "2024-11-05", "in_use", "E-310", True, ""),
        ("D-4010", "laptop", tp, "2022-01-20", "in_use", "E-311", True, ""),
        ("D-4011", "laptop", mac, "2021-06-01", "in_use", "E-312", True, ""),
        ("D-4012", "laptop", tp, "2024-04-04", "in_use", "E-313", True, ""),
        ("D-4013", "laptop", tp, "2025-02-02", "in_use", "E-314", True, ""),
        ("D-4014", "laptop", tp, "2023-03-03", "in_use", "E-315", True, ""),
        ("D-4015", "laptop", tp, "2024-07-07", "in_use", "E-317", True, ""),
        ("D-4016", "laptop", tp, "2021-09-15", "in_use", "E-318", True, ""),
        ("D-4017", "laptop", tp, "2022-11-16", "in_use", "E-319", True, ""),
        ("D-4018", "laptop", tp, "2024-10-10", "in_use", "E-320", True, ""),
        ("D-4019", "laptop", tp, "2020-12-01", "in_use", "E-321", True, ""),
        ("D-4020", "laptop", tp, "2022-03-08", "in_use", "E-322", True, ""),
        ("D-4021", "laptop", tp, "2022-08-30", "in_use", "E-323", True, ""),
        ("D-4022", "laptop", tp, "2025-05-05", "in_use", "E-324", True, ""),
        ("D-4023", "laptop", tp, "2021-11-11", "in_use", "E-325", True, ""),
        ("D-4024", "laptop", tp, "2023-12-12", "in_use", "E-326", True, ""),
        ("D-4025", "laptop", mac, "2024-02-02", "in_use", "E-327", True, ""),
        ("D-4026", "laptop", tp, "2023-06-06", "in_use", "E-331", True, ""),
        ("D-4027", "laptop", tp, "2025-08-08", "in_use", "E-333", True, ""),
        ("D-4030", "laptop", tp + " Gen 6", "2026-09-01", "in_stock", "", True, ""),
        ("D-4031", "laptop", tp + " Gen 6", "2026-10-20", "in_stock", "", False, ""),
        ("D-4032", "laptop", mac, "2026-06-15", "in_stock", "", True, ""),
        ("D-4033", "laptop", tp, "2026-03-01", "in_stock", "", True, ""),
        ("D-4034", "laptop", tp + " Gen 6", "2026-10-01", "repair", "", True, "E-310"),
        ("D-4035", "laptop", tp, "2021-04-04", "in_stock", "", True, "E-316"),
        ("D-4036", "laptop", mac, "2025-11-20", "in_stock", "", False, "E-307"),
        ("D-4040", "laptop", tp, "2025-02-20", "repair", "", True, "E-330"),
        ("D-4045", "laptop", tp, "2024-01-10", "repair", "", True, "E-331"),
        ("D-4046", "laptop", mac, "2021-08-08", "repair", "", True, "E-326"),
        ("D-4047", "laptop", tp, "2024-05-05", "repair", "", True, "E-332"),
        ("D-5001", "monitor", "Dell P2419 24-inch", "2021-05-05", "in_use", "E-301", False, ""),
        ("D-5002", "monitor", "Dell U2723 27-inch", "2023-07-07", "in_use", "E-302", False, ""),
        ("D-5003", "monitor", "Dell P2419 24-inch", "2020-02-02", "in_use", "E-305", False, ""),
        ("D-5010", "monitor", "Dell U2723 27-inch", "2023-04-04", "in_use", "E-305", False, ""),
        ("D-5020", "monitor", "Dell U2723 27-inch", "2022-09-09", "in_use", "E-312", False, ""),
        ("D-5021", "monitor", "Dell P2419 24-inch", "2021-10-10", "in_use", "E-312", False, ""),
        ("D-5022", "monitor", "Dell P2419 24-inch", "2022-02-22", "in_use", "E-313", False, ""),
        ("D-5030", "monitor", "Dell U2723 27-inch", "2025-03-03", "in_stock", "", False, ""),
        ("D-5031", "monitor", "LG 27UL850 27-inch", "2024-09-09", "repair", "", False, "E-302"),
        ("D-5032", "monitor", "LG 27UL850 27-inch", "2025-09-15", "in_stock", "", False, ""),
        ("D-5033", "monitor", "Dell U2723 27-inch", "2026-02-02", "repair", "", False, ""),
        ("D-5034", "monitor", "Dell P2424 24-inch", "2026-05-05", "in_stock", "", False, ""),
        ("D-5035", "monitor", "Samsung S27 27-inch", "2025-06-01", "in_stock", "", False, ""),
        ("D-5036", "monitor", "Dell P2419 24-inch", "2022-06-06", "in_use", "E-330", False, ""),
        ("D-5037", "monitor", "Dell U2723 27-inch", "2024-01-01", "in_use", "E-320", False, ""),
        ("D-6001", "phone", "iPhone 15", "2024-02-12", "in_use", "E-312", True, ""),
        ("D-6002", "phone", "iPhone 15", "2024-03-01", "in_use", "E-313", True, ""),
        ("D-6003", "phone", "Pixel 8", "2024-05-20", "in_use", "E-325", True, ""),
        ("D-6004", "phone", "Pixel 9", "2025-10-01", "in_stock", "", True, ""),
        ("D-6012", "phone", "iPhone 16", "2026-01-15", "in_stock", "", True, "E-326"),
    ]:
        devs[did] = {"kind": kind, "model": model, "purchased": bought, "status": status, "assigned_to": who,
                     "encrypted": enc, "last_user": last}
    lic = {}
    for lid, product, seats, allowed in [
        ("SW-11", "Figma Professional", 14, ["Design"]), ("SW-12", "Figma Organization", 6, []),
        ("SW-21", "Adobe Acrobat Pro", 10, []), ("SW-22", "Adobe Acrobat Standard", 12, []),
        ("SW-23", "Adobe Creative Cloud", 6, ["Design"]), ("SW-31", "JetBrains All Products", 8, ["Engineering"]),
        ("SW-32", "GitHub Enterprise", 10, ["Engineering", "Support"]), ("SW-41", "Zendesk Agent", 8, ["Support"]),
        ("SW-52", "Miro", 10, []),
    ]:
        lic[lid] = {"product": product, "seats": seats, "allowed_depts": allowed}
    seats = {}
    n = [100]

    def seat(l, e, used, status="active", assigned="2025-01-06"):
        n[0] += 1
        seats["SE-%d" % n[0]] = {"licence": l, "employee": e, "assigned": assigned,
                                 "last_used": "2026-" + used if used else "", "status": status}
    for e, u in [("E-301", "11-13"), ("E-302", "11-10"), ("E-303", "08-02"), ("E-304", "09-17"), ("E-305", "11-14"),
                 ("E-307", "10-01"), ("E-308", "09-16"), ("E-309", "10-30"), ("E-312", "11-12"), ("E-325", "09-01"),
                 ("E-310", "10-15")]:
        seat("SW-11", e, u)
    seat("SW-11", "E-326", "07-07", "revoked")
    seat("SW-11", "E-327", "11-02")
    for e, u in [("E-303", "05-05"), ("E-315", "11-01"), ("E-316", "06-01")]:
        seat("SW-12", e, u)
    for e, u in [("E-301", "11-11"), ("E-302", "11-09"), ("E-307", "09-20"), ("E-312", "11-13"), ("E-308", "10-28"),
                 ("E-303", "11-02")]:
        seat("SW-23", e, u)
    for e, u in [("E-310", "11-12"), ("E-311", "11-03"), ("E-315", "10-30"), ("E-316", "08-20"), ("E-317", "11-15"),
                 ("E-320", "11-05"), ("E-325", "06-30"), ("E-326", "11-01"), ("E-327", "10-10")]:
        seat("SW-21", e, u)
    for e, u in [("E-318", "11-12"), ("E-319", "11-13"), ("E-321", "11-02"), ("E-322", "10-21"), ("E-323", "11-09"),
                 ("E-324", "11-14"), ("E-314", "11-10"), ("E-328", "09-15"), ("E-331", "11-11"), ("E-313", "11-04"),
                 ("E-306", "08-08")]:
        seat("SW-22", e, u)
    for e, u in [("E-310", "11-15"), ("E-311", "11-14"), ("E-315", "11-13"), ("E-316", "08-25"), ("E-317", "11-12"),
                 ("E-320", "11-13"), ("E-330", "11-01")]:
        seat("SW-31", e, u)
    for e, u in [("E-310", "11-15"), ("E-311", "11-10"), ("E-315", "11-12"), ("E-317", "11-14"), ("E-320", "11-13"),
                 ("E-330", "10-29"), ("E-318", "11-02")]:
        seat("SW-32", e, u)
    for e, u in [("E-318", "11-15"), ("E-319", "11-14"), ("E-321", "11-13"), ("E-322", "11-12"), ("E-323", "11-10"),
                 ("E-325", "09-09"), ("E-313", "10-01"), ("E-335", "10-08")]:
        seat("SW-41", e, u)
    for e, u in [("E-301", "11-14"), ("E-302", "11-03"), ("E-303", "10-12"), ("E-305", "11-10"), ("E-308", "11-01"),
                 ("E-312", "11-12"), ("E-320", "11-04"), ("E-326", "09-30"), ("E-327", "11-13")]:
        seat("SW-52", e, u)
    st = {"today": "2026-11-16",
          "policy": {"laptop_max_age_years": 4, "inactivity_days": 60,
                     "max_devices": {"laptop": 1, "monitor": 2, "phone": 1},
                     "bundles": {"Design": ["Figma Professional", "Adobe Creative Cloud", "Miro"],
                                 "Engineering": ["JetBrains All Products", "GitHub Enterprise"],
                                 "Support": ["Zendesk Agent", "Adobe Acrobat Standard"]}},
          "employees": emps, "devices": devs, "licences": lic, "seats": seats, "next_seat": 200}
    # sanity
    for k, l in lic.items():
        assert len([1 for x in seats.values() if x["licence"] == k and x["status"] == "active"]) <= l["seats"], k
    for e in emps:
        for kind, cap in st["policy"]["max_devices"].items():
            assert len([1 for d in devs.values() if d["assigned_to"] == e and d["kind"] == kind]) <= cap, (e, kind)
    for k, d in devs.items():
        assert (d["status"] == "in_use") == bool(d["assigned_to"]), k
    return st


def _seat_id(i, lic, emp):
    ks = [k for k, x in i["seats"].items() if x["licence"] == lic and x["employee"] == emp and x["status"] == "active"]
    assert len(ks) == 1, (lic, emp)
    return ks[0]


def _revoked(i, pairs):
    return dict((_seat_id(i, l, e), {"status": "revoked"}) for l, e in pairs)


def _newseat(lic, emp):
    return {"licence": lic, "employee": emp, "assigned": "2026-11-16", "last_used": "", "status": "active"}


_add("T09", "assets", _assets(),
     "Marcus Hale's last day is today, so please do his offboarding. Take back every device he has: his laptop "
     "goes back into stock, unless it is past the age limit in our policy, in which case it is retired instead; "
     "monitors go back into stock; his phone goes to repair for a battery check before anyone else gets it. Take "
     "back all of his software seats, and when he holds nothing any more, mark him as having left. Careful: Marcus "
     "Hales in Sales is leaving next month and Marcia Hale is staying - don't touch either of them.",
     [["get_it_policy", {}], ["find_employees", {"query": "Marcus Hale"}], ["get_employee", {"employee_id": "E-312"}],
      ["unassign_device", {"device_id": "D-4011", "destination": "in_stock"}], ["retire_device", {"device_id": "D-4011"}],
      ["unassign_device", {"device_id": "D-5020", "destination": "in_stock"}],
      ["unassign_device", {"device_id": "D-5021", "destination": "in_stock"}],
      ["unassign_device", {"device_id": "D-6001", "destination": "repair"}],
      ["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-312"}],
      ["revoke_seat", {"licence_id": "SW-23", "employee_id": "E-312"}],
      ["revoke_seat", {"licence_id": "SW-52", "employee_id": "E-312"}],
      ["update_employee", {"employee_id": "E-312", "status": "left"}]],
     [["unassign_device", {"device_id": "D-4011", "destination": "in_stock"}],
      ["unassign_device", {"device_id": "D-5020", "destination": "in_stock"}],
      ["unassign_device", {"device_id": "D-5021", "destination": "in_stock"}],
      ["unassign_device", {"device_id": "D-6001", "destination": "repair"}],
      ["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-312"}],
      ["revoke_seat", {"licence_id": "SW-23", "employee_id": "E-312"}],
      ["revoke_seat", {"licence_id": "SW-52", "employee_id": "E-312"}],
      ["update_employee", {"employee_id": "E-312", "status": "left"}]],
     lambda s, i: _final(s, i, changes={
         "devices": {"D-4011": {"status": "retired", "assigned_to": "", "last_user": "E-312"},
                     "D-5020": {"status": "in_stock", "assigned_to": "", "last_user": "E-312"},
                     "D-5021": {"status": "in_stock", "assigned_to": "", "last_user": "E-312"},
                     "D-6001": {"status": "repair", "assigned_to": "", "last_user": "E-312"}},
         "seats": _revoked(i, [("SW-11", "E-312"), ("SW-23", "E-312"), ("SW-52", "E-312")]),
         "employees": {"E-312": {"status": "left"}}}),
     mentions=["Marcus Hale", "Marcus Hales", "Marcia Hale"])

_add("T10", "assets", _assets(),
     "The Figma Professional renewal is coming up and I want to trim it. Take back every Figma Professional seat "
     "that has been inactive for longer than our policy allows, and also every seat held by someone who has already "
     "left the company, however recently they used it. Then set the licence's seat count to the number of seats "
     "still in use plus 2 spare. Figma Organization is a separate licence and is not part of this; nothing else "
     "should change.",
     [["get_it_policy", {}], ["list_licences", {}], ["list_seats", {"licence_id": "SW-11"}],
      ["list_seats", {"licence_id": "SW-11", "page": 2}],
      ["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-303"}],
      ["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-307"}],
      ["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-308"}],
      ["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-325"}],
      ["set_seat_count", {"licence_id": "SW-11", "seats": 10}]],
     [["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-303"}],
      ["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-307"}],
      ["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-325"}],
      ["set_seat_count", {"licence_id": "SW-11", "seats": 11}]],
     lambda s, i: _final(s, i, changes={
         "seats": _revoked(i, [("SW-11", "E-303"), ("SW-11", "E-307"), ("SW-11", "E-308"), ("SW-11", "E-325")]),
         "licences": {"SW-11": {"seats": 10}}}),
     mentions=["Figma Professional", "Figma Organization"])


def _c11(s, i):
    opts = []
    for a, b in (("D-4030", "D-4032"), ("D-4032", "D-4030")):
        opts.append(({"devices": {a: {"status": "in_use", "assigned_to": "E-340"},
                                  b: {"status": "in_use", "assigned_to": "E-341"}},
                      "licences": {"SW-23": {"seats": 8}, "SW-52": {"seats": 11}}},
                     {"seats": [_newseat(l, e) for l in ("SW-11", "SW-23", "SW-52") for e in ("E-340", "E-341")]}))
    return _any(*opts)(s, i)


_add("T11", "assets", _assets(),
     "Two designers start today, Noemi Laine and Theo Brandvold. Set each of them up: one laptop each - give them "
     "the most recently purchased laptops in stock that policy allows us to hand out - and a seat on every product "
     "in the Design starter bundle. Where a licence doesn't have enough free seats for both of them, raise its seat "
     "count by exactly the number missing, no more. Don't take seats away from anyone, and leave the other starter "
     "and everything else as it is.",
     [["get_it_policy", {}], ["find_employees", {"status": "starting"}],
      ["list_devices", {"kind": "laptop", "status": "in_stock"}], ["list_licences", {}],
      ["assign_device", {"device_id": "D-4030", "employee_id": "E-340"}],
      ["assign_device", {"device_id": "D-4032", "employee_id": "E-341"}],
      ["assign_seat", {"licence_id": "SW-11", "employee_id": "E-340"}],
      ["assign_seat", {"licence_id": "SW-11", "employee_id": "E-341"}],
      ["set_seat_count", {"licence_id": "SW-23", "seats": 8}],
      ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-340"}],
      ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-341"}],
      ["set_seat_count", {"licence_id": "SW-52", "seats": 11}],
      ["assign_seat", {"licence_id": "SW-52", "employee_id": "E-340"}],
      ["assign_seat", {"licence_id": "SW-52", "employee_id": "E-341"}]],
     [["assign_device", {"device_id": "D-4031", "employee_id": "E-340"}],
      ["assign_device", {"device_id": "D-4030", "employee_id": "E-340"}],
      ["assign_device", {"device_id": "D-4032", "employee_id": "E-341"}],
      ["assign_seat", {"licence_id": "SW-11", "employee_id": "E-340"}],
      ["assign_seat", {"licence_id": "SW-11", "employee_id": "E-341"}],
      ["set_seat_count", {"licence_id": "SW-23", "seats": 8}],
      ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-340"}],
      ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-341"}],
      ["set_seat_count", {"licence_id": "SW-52", "seats": 12}],
      ["assign_seat", {"licence_id": "SW-52", "employee_id": "E-340"}],
      ["assign_seat", {"licence_id": "SW-52", "employee_id": "E-341"}]],
     _c11, mentions=["Noemi Laine", "Theo Brandvold", "Design"])

_add("T12", "assets", _assets(),
     "Ana Ferreira in Design has asked for an extra monitor: the newest 27-inch monitor we have in stock. If our "
     "policy lets her have one more monitor, just give it to her. If it doesn't, swap instead: her older monitor "
     "(the one purchased earlier) comes back into stock and she gets the new one in its place. Don't touch her "
     "laptop or anything else, and make sure you're dealing with Ana Ferreira and not Ana Ferreira-Lopes from Sales.",
     [["get_it_policy", {}], ["find_employees", {"query": "Ana Ferreira"}], ["get_employee", {"employee_id": "E-305"}],
      ["list_devices", {"kind": "monitor", "status": "in_stock"}],
      ["assign_device", {"device_id": "D-5032", "employee_id": "E-305"}],
      ["unassign_device", {"device_id": "D-5003", "destination": "in_stock"}],
      ["assign_device", {"device_id": "D-5032", "employee_id": "E-305"}]],
     [["unassign_device", {"device_id": "D-5003", "destination": "in_stock"}],
      ["assign_device", {"device_id": "D-5034", "employee_id": "E-305"}]],
     lambda s, i: _final(s, i, changes={"devices": {
         "D-5003": {"status": "in_stock", "assigned_to": "", "last_user": "E-305"},
         "D-5032": {"status": "in_use", "assigned_to": "E-305"}}}),
     ref_errors=[4], mentions=["Ana Ferreira", "Ana Ferreira-Lopes", "Design"])

_add("T13", "assets", _assets(),
     "Sven Aalto moves from Engineering to Design today. Update his department, and sort out his software so it "
     "matches the move: he loses every seat that Design people aren't allowed to hold, keeps the ones they are "
     "allowed, and gets a seat on every Design starter-bundle product he doesn't have yet. If a licence he needs "
     "has no free seat, free one up by taking back a seat from someone who has already left the company rather than "
     "raising the seat count. His laptop and monitor stay with him, and nothing else should change.",
     [["get_it_policy", {}], ["find_employees", {"query": "Sven Aalto"}], ["get_employee", {"employee_id": "E-320"}],
      ["list_licences", {}], ["update_employee", {"employee_id": "E-320", "dept": "Design"}],
      ["revoke_seat", {"licence_id": "SW-31", "employee_id": "E-320"}],
      ["revoke_seat", {"licence_id": "SW-32", "employee_id": "E-320"}],
      ["update_employee", {"employee_id": "E-320", "dept": "Design"}],
      ["assign_seat", {"licence_id": "SW-11", "employee_id": "E-320"}],
      ["list_seats", {"licence_id": "SW-23"}],
      ["revoke_seat", {"licence_id": "SW-23", "employee_id": "E-307"}],
      ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-320"}]],
     [["revoke_seat", {"licence_id": "SW-31", "employee_id": "E-320"}],
      ["revoke_seat", {"licence_id": "SW-32", "employee_id": "E-320"}],
      ["update_employee", {"employee_id": "E-320", "dept": "Design"}],
      ["assign_seat", {"licence_id": "SW-11", "employee_id": "E-320"}],
      ["set_seat_count", {"licence_id": "SW-23", "seats": 7}],
      ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-320"}]],
     lambda s, i: _final(s, i, changes={
         "employees": {"E-320": {"dept": "Design"}},
         "seats": _revoked(i, [("SW-31", "E-320"), ("SW-32", "E-320"), ("SW-23", "E-307")])},
         new={"seats": [_newseat("SW-11", "E-320"), _newseat("SW-23", "E-320")]}),
     ref_errors=[4], mentions=["Sven Aalto", "Engineering", "Design"])


def _c14(s, i):
    olds = {"E-321": "D-4019", "E-318": "D-4016", "E-322": "D-4020"}
    opts = []
    for perm in itertools.permutations(["D-4030", "D-4032", "D-4033"]):
        ch = {}
        for (emp, old), new in zip(sorted(olds.items()), perm):
            ch[old] = {"status": "retired", "assigned_to": "", "last_user": emp}
            ch[new] = {"status": "in_use", "assigned_to": emp}
        opts.append(({"devices": ch}, None))
    return _any(*opts)(s, i)


_add("T14", "assets", _assets(),
     "Time to refresh the end-of-life laptops in the Support team. Every Support person whose laptop is past the "
     "age limit in our policy gets a replacement from stock - only laptops that policy allows us to hand out - and "
     "their old laptop is retired. We may not have enough suitable stock for everyone; if so, replace the oldest "
     "laptops first and leave the remaining people on their current laptop for now. Other teams' laptops are not "
     "part of this, and nothing else should change.",
     [["get_it_policy", {}], ["find_employees", {"dept": "Support"}],
      ["list_devices", {"kind": "laptop", "status": "in_use"}],
      ["list_devices", {"kind": "laptop", "status": "in_use", "page": 2}],
      ["list_devices", {"kind": "laptop", "status": "in_use", "page": 3}],
      ["list_devices", {"kind": "laptop", "status": "in_stock"}],
      ["unassign_device", {"device_id": "D-4019", "destination": "in_stock"}],
      ["retire_device", {"device_id": "D-4019"}],
      ["assign_device", {"device_id": "D-4030", "employee_id": "E-321"}],
      ["unassign_device", {"device_id": "D-4016", "destination": "in_stock"}],
      ["retire_device", {"device_id": "D-4016"}],
      ["assign_device", {"device_id": "D-4032", "employee_id": "E-318"}],
      ["unassign_device", {"device_id": "D-4020", "destination": "in_stock"}],
      ["retire_device", {"device_id": "D-4020"}],
      ["assign_device", {"device_id": "D-4033", "employee_id": "E-322"}]],
     [["unassign_device", {"device_id": "D-4019", "destination": "in_stock"}],
      ["retire_device", {"device_id": "D-4019"}],
      ["assign_device", {"device_id": "D-4030", "employee_id": "E-321"}],
      ["unassign_device", {"device_id": "D-4016", "destination": "in_stock"}],
      ["retire_device", {"device_id": "D-4016"}],
      ["assign_device", {"device_id": "D-4032", "employee_id": "E-318"}],
      ["unassign_device", {"device_id": "D-4021", "destination": "in_stock"}],
      ["retire_device", {"device_id": "D-4021"}],
      ["assign_device", {"device_id": "D-4033", "employee_id": "E-323"}]],
     _c14, mentions=["Support"])

_add("T15", "assets", _assets(),
     "Adobe true-up. For each of our Adobe licences, first take back the seats held by people who have already "
     "left the company (people who are still working out their notice keep theirs). Then set each Adobe licence's "
     "seat count to the number of seats in use plus 10%, rounded up to a whole seat; where that is already the "
     "seat count, leave the count alone. Inactive seats of current staff are not part of this exercise, and "
     "non-Adobe licences stay exactly as they are.",
     [["list_licences", {}], ["list_seats", {"licence_id": "SW-21"}], ["list_seats", {"licence_id": "SW-21", "page": 2}],
      ["list_seats", {"licence_id": "SW-22"}], ["list_seats", {"licence_id": "SW-22", "page": 2}],
      ["list_seats", {"licence_id": "SW-23"}],
      ["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-316"}],
      ["revoke_seat", {"licence_id": "SW-22", "employee_id": "E-328"}],
      ["revoke_seat", {"licence_id": "SW-23", "employee_id": "E-307"}],
      ["set_seat_count", {"licence_id": "SW-21", "seats": 9}],
      ["set_seat_count", {"licence_id": "SW-22", "seats": 11}]],
     [["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-316"}],
      ["revoke_seat", {"licence_id": "SW-22", "employee_id": "E-328"}],
      ["revoke_seat", {"licence_id": "SW-22", "employee_id": "E-313"}],
      ["revoke_seat", {"licence_id": "SW-23", "employee_id": "E-307"}],
      ["set_seat_count", {"licence_id": "SW-21", "seats": 9}],
      ["set_seat_count", {"licence_id": "SW-22", "seats": 12}]],
     lambda s, i: _final(s, i, changes={
         "seats": _revoked(i, [("SW-21", "E-316"), ("SW-22", "E-328"), ("SW-23", "E-307")]),
         "licences": {"SW-21": {"seats": 9}, "SW-22": {"seats": 11}}}),
     mentions=["Adobe"])

_add("T16", "assets", _assets(),
     "These devices came back from the repair shop today: D-4040, D-4045, D-4046, D-4047, D-5031 and D-6012. Put "
     "each one that was actually out for repair back into stock. Exception: anything past the policy age limit is "
     "retired instead. Then, for each laptop that went back into stock, if the person who last used it still works "
     "here and has no laptop at the moment, give it straight back to them. A device on the list that wasn't in "
     "repair needs nothing, and other devices in repair stay there. Nothing else should change.",
     [["get_it_policy", {}], ["list_devices", {"status": "repair"}],
      ["get_employee", {"employee_id": "E-330"}], ["get_employee", {"employee_id": "E-331"}],
      ["get_employee", {"employee_id": "E-332"}],
      ["set_device_status", {"device_id": "D-4040", "status": "in_stock"}],
      ["assign_device", {"device_id": "D-4040", "employee_id": "E-330"}],
      ["set_device_status", {"device_id": "D-4045", "status": "in_stock"}],
      ["retire_device", {"device_id": "D-4046"}],
      ["set_device_status", {"device_id": "D-4047", "status": "in_stock"}],
      ["set_device_status", {"device_id": "D-5031", "status": "in_stock"}]],
     [["set_device_status", {"device_id": "D-4040", "status": "in_stock"}],
      ["assign_device", {"device_id": "D-4040", "employee_id": "E-330"}],
      ["set_device_status", {"device_id": "D-4045", "status": "in_stock"}],
      ["set_device_status", {"device_id": "D-4046", "status": "in_stock"}],
      ["set_device_status", {"device_id": "D-4047", "status": "in_stock"}],
      ["set_device_status", {"device_id": "D-5031", "status": "in_stock"}]],
     lambda s, i: _final(s, i, changes={"devices": {
         "D-4040": {"status": "in_use", "assigned_to": "E-330"}, "D-4045": {"status": "in_stock"},
         "D-4046": {"status": "retired"}, "D-4047": {"status": "in_stock"}, "D-5031": {"status": "in_stock"}}}))


# --------------------------------------------------------------------------
# 3. meal-kit kitchen: recipes, stock lots (FEFO), production batches, lines
# --------------------------------------------------------------------------
_NUTS = ("peanut", "tree_nut")


class KitchenEnv(_Env):
    TOOLS = [
        _fn("get_kitchen_info", "Today's date, the production lines (daily capacity in portions, nut-free flag) and "
            "the planning and stock rules that the write tools enforce."),
        _fn("search_recipes", "Search recipes by name (case-insensitive substring).",
            {"query": _s("Text that must appear in the recipe name.")}, ["query"]),
        _fn("get_recipe", "One recipe: grams of each ingredient per portion and its allergens.",
            {"recipe_id": _s("Recipe id, e.g. RC-01.")}, ["recipe_id"]),
        _fn("find_ingredients", "Search ingredients by name (case-insensitive substring).",
            {"query": _s("Text that must appear in the ingredient name.")}, ["query"]),
        _fn("list_lots", "Stock lots, filtered, earliest expiry first; 8 per page. 'free' is the quantity not yet "
            "allocated to planned batches.",
            {"ingredient_id": _s("Only lots of this ingredient, e.g. ING-01."),
             "status": _s("Only lots in this status.", ["available", "quarantined"]),
             "expires_before": _s("Only lots expiring before this date, YYYY-MM-DD."), "page": _PAGE}),
        _fn("list_batches", "Production batches, filtered, by date, line and id; 8 per page.",
            {"date": _s("Only this production date, YYYY-MM-DD."), "line": _s("Only this line.", ["L1", "L2", "L3"]),
             "recipe_id": _s("Only batches of this recipe."),
             "status": _s("Only this status.", ["planned", "released", "cancelled"]), "page": _PAGE}),
        _fn("get_batch", "One batch: recipe, portions, date, line, status, and per ingredient the grams needed, "
            "allocated (by lot) and still missing.", {"batch_id": _s("Batch id, e.g. PB-801.")}, ["batch_id"]),
        _fn("create_batch", "Plan a new production batch (status planned, nothing allocated).",
            {"recipe_id": _s("Recipe id."), "portions": _i("Number of portions.", 1),
             "date": _s("Production date, YYYY-MM-DD."), "line": _s("Production line.", ["L1", "L2", "L3"])},
            ["recipe_id", "portions", "date", "line"]),
        _fn("update_batch", "Change the portions and/or line of a planned batch.",
            {"batch_id": _s("Batch id."), "portions": _i("New number of portions.", 1),
             "line": _s("New production line.", ["L1", "L2", "L3"])}, ["batch_id"]),
        _fn("cancel_batch", "Cancel a planned batch; its allocations are released back to stock.",
            {"batch_id": _s("Batch id.")}, ["batch_id"]),
        _fn("allocate_lot", "Allocate grams from a stock lot to a planned batch.",
            {"batch_id": _s("Batch id."), "lot_id": _s("Lot id, e.g. LOT-0103."), "grams": _i("Grams to allocate.", 1)},
            ["batch_id", "lot_id", "grams"]),
        _fn("deallocate_lot", "Remove the whole allocation of one lot from a planned batch.",
            {"batch_id": _s("Batch id."), "lot_id": _s("Lot id.")}, ["batch_id", "lot_id"]),
        _fn("release_batch", "Release a fully allocated planned batch to production (its allocated stock is consumed).",
            {"batch_id": _s("Batch id.")}, ["batch_id"]),
        _fn("quarantine_lot", "Quarantine a stock lot so it can no longer be used.",
            {"lot_id": _s("Lot id."), "reason": _s("Reason.", ["recall", "expired", "damaged"])}, ["lot_id", "reason"]),
    ]

    def _allergens(self, rid):
        out = set()
        for ing in self.state["recipes"][rid]["per_portion"]:
            out.update(self.state["ingredients"][ing]["allergens"])
        return sorted(out)

    def _used(self, date, line, exclude=None):
        return sum(b["portions"] for k, b in self.state["batches"].items()
                   if b["date"] == date and b["line"] == line and b["status"] in ("planned", "released") and k != exclude)

    def _free(self, lot):
        return self.state["lots"][lot]["qty"] - sum(b["alloc"].get(lot, 0) for b in self.state["batches"].values()
                                                    if b["status"] == "planned")

    def _need(self, b):
        return dict((ing, g * b["portions"]) for ing, g in self.state["recipes"][b["recipe"]]["per_portion"].items())

    def _alloc_by_ing(self, b):
        out = {}
        for lot, g in b["alloc"].items():
            ing = self.state["lots"][lot]["ingredient"]
            out[ing] = out.get(ing, 0) + g
        return out

    def _eligible(self, lot, b):
        x = self.state["lots"][lot]
        return x["status"] == "available" and x["expires"] >= b["date"]

    def _check_line(self, rid, date, line, portions, exclude=None):
        ln = self.state["lines"][line]
        if ln["nut_free"] and set(self._allergens(rid)) & set(_NUTS):
            raise _Rule("%s contains %s and cannot run on nut-free line %s" % (
                self.state["recipes"][rid]["name"], "/".join(sorted(set(self._allergens(rid)) & set(_NUTS))), line))
        used = self._used(date, line, exclude)
        if used + portions > ln["capacity"]:
            raise _Rule("line %s has %d of %d portions booked on %s; %d more do not fit" % (
                line, used, ln["capacity"], date, portions))

    def _batch_row(self, k):
        b = self.state["batches"][k]
        need, have = self._need(b), self._alloc_by_ing(b)
        return {"batch_id": k, "recipe_id": b["recipe"], "recipe": self.state["recipes"][b["recipe"]]["name"],
                "portions": b["portions"], "date": b["date"], "line": b["line"], "status": b["status"],
                "fully_allocated": all(have.get(i, 0) == g for i, g in need.items())}

    def _t_get_kitchen_info(self, a):
        return {"today": self.state["today"], "lines": [dict(line=k, **v) for k, v in sorted(self.state["lines"].items())],
                "rules": [
                    "Batches can only be planned for today or later. The portions of all planned and released "
                    "batches on a line on one day may not exceed the line's capacity.",
                    "Nut-free lines may not run recipes containing peanut or tree_nut. Coconut is not classed as a "
                    "tree nut. Sesame is not a nut.",
                    "Only available (not quarantined) lots may be allocated, and only to a batch whose date is on or "
                    "before the lot's expiry date.",
                    "First-expiry-first-out: for each ingredient, a batch must take stock from the earliest-expiring "
                    "lot that it may use and that still has free stock (ties: lower lot id) before any later lot.",
                    "A batch may never be allocated more of an ingredient than it needs (portions x grams per "
                    "portion). Reducing portions is only possible while allocations still fit the new need.",
                    "A batch can be released only when every ingredient is allocated exactly. Released batches "
                    "cannot be changed, cancelled or re-allocated.",
                    "A lot cannot be quarantined while it is allocated to a planned batch; take the allocation off "
                    "first.",
                ]}

    def _t_search_recipes(self, a):
        q = a["query"].strip().lower()
        if not q:
            raise _Rule("query must not be empty")
        return {"recipes": [{"recipe_id": k, "name": r["name"], "status": r["status"], "allergens": self._allergens(k)}
                            for k, r in sorted(self.state["recipes"].items()) if q in r["name"].lower()]}

    def _t_get_recipe(self, a):
        r = self._get("recipes", a["recipe_id"], "recipe")
        return {"recipe_id": a["recipe_id"], "name": r["name"], "status": r["status"],
                "allergens": self._allergens(a["recipe_id"]),
                "per_portion": [{"ingredient_id": ing, "ingredient": self.state["ingredients"][ing]["name"], "grams": g}
                                for ing, g in sorted(r["per_portion"].items())]}

    def _t_find_ingredients(self, a):
        q = a["query"].strip().lower()
        if not q:
            raise _Rule("query must not be empty")
        return {"ingredients": [{"ingredient_id": k, "name": x["name"], "allergens": x["allergens"]}
                                for k, x in sorted(self.state["ingredients"].items()) if q in x["name"].lower()]}

    def _t_list_lots(self, a):
        if "ingredient_id" in a:
            self._get("ingredients", a["ingredient_id"], "ingredient")
        if "expires_before" in a:
            _pdate(a["expires_before"], "expires_before")
        rows = [k for _, k in sorted((x["expires"], k) for k, x in self.state["lots"].items()
                                     if ("ingredient_id" not in a or x["ingredient"] == a["ingredient_id"])
                                     and ("status" not in a or x["status"] == a["status"])
                                     and ("expires_before" not in a or x["expires"] < a["expires_before"]))]
        meta, rows = _paged(rows, a.get("page", 1), 8)
        out = []
        for k in rows:
            x = self.state["lots"][k]
            out.append({"lot_id": k, "ingredient_id": x["ingredient"],
                        "ingredient": self.state["ingredients"][x["ingredient"]]["name"], "qty": x["qty"],
                        "free": self._free(k), "expires": x["expires"], "supplier": x["supplier"],
                        "status": x["status"], "reason": x["reason"],
                        "allocated_to": dict((bk, b["alloc"][k]) for bk, b in sorted(self.state["batches"].items())
                                             if b["status"] == "planned" and k in b["alloc"])})
        meta["lots"] = out
        return meta

    def _t_list_batches(self, a):
        if "date" in a:
            _pdate(a["date"])
        if "recipe_id" in a:
            self._get("recipes", a["recipe_id"], "recipe")
        rows = [k for _, _, k in sorted((b["date"], b["line"], k) for k, b in self.state["batches"].items()
                                        if ("date" not in a or b["date"] == a["date"])
                                        and ("line" not in a or b["line"] == a["line"])
                                        and ("recipe_id" not in a or b["recipe"] == a["recipe_id"])
                                        and ("status" not in a or b["status"] == a["status"]))]
        meta, rows = _paged(rows, a.get("page", 1), 8)
        meta["batches"] = [self._batch_row(k) for k in rows]
        return meta

    def _t_get_batch(self, a):
        b = self._get("batches", a["batch_id"], "batch")
        row = self._batch_row(a["batch_id"])
        need, have = self._need(b), self._alloc_by_ing(b)
        row["ingredients"] = [{"ingredient_id": ing, "ingredient": self.state["ingredients"][ing]["name"],
                               "needed": g, "allocated": have.get(ing, 0), "missing": g - have.get(ing, 0),
                               "lots": dict((l, q) for l, q in sorted(b["alloc"].items())
                                            if self.state["lots"][l]["ingredient"] == ing)}
                              for ing, g in sorted(need.items())]
        row["allergens"] = self._allergens(b["recipe"])
        return row

    def _t_create_batch(self, a):
        r = self._get("recipes", a["recipe_id"], "recipe")
        if r["status"] != "active":
            raise _Rule("recipe %s is %s" % (a["recipe_id"], r["status"]))
        if _pdate(a["date"]) < _pdate(self.state["today"]):
            raise _Rule("cannot plan a batch in the past (today is %s)" % self.state["today"])
        self._check_line(a["recipe_id"], a["date"], a["line"], a["portions"])
        bid = self._new_id("next_batch", "PB")
        self.state["batches"][bid] = {"recipe": a["recipe_id"], "portions": a["portions"], "date": a["date"],
                                      "line": a["line"], "status": "planned", "alloc": {}}
        return {"ok": True, "batch": self._batch_row(bid)}

    def _planned(self, bid):
        b = self._get("batches", bid, "batch")
        if b["status"] != "planned":
            raise _Rule("batch %s is %s, not planned" % (bid, b["status"]))
        return b

    def _t_update_batch(self, a):
        b = self._planned(a["batch_id"])
        if "portions" not in a and "line" not in a:
            raise _Rule("give portions and/or line")
        portions = a.get("portions", b["portions"])
        line = a.get("line", b["line"])
        if portions == b["portions"] and line == b["line"]:
            raise _Rule("nothing to change")
        self._check_line(b["recipe"], b["date"], line, portions, exclude=a["batch_id"])
        have = self._alloc_by_ing(b)
        for ing, g in self.state["recipes"][b["recipe"]]["per_portion"].items():
            if have.get(ing, 0) > g * portions:
                raise _Rule("batch %s already has %d g of %s allocated, more than %d portions need (%d g); "
                            "deallocate first" % (a["batch_id"], have[ing], ing, portions, g * portions))
        b["portions"], b["line"] = portions, line
        return {"ok": True, "batch": self._batch_row(a["batch_id"])}

    def _t_cancel_batch(self, a):
        b = self._planned(a["batch_id"])
        b["status"], b["alloc"] = "cancelled", {}
        return {"ok": True, "batch_id": a["batch_id"], "status": "cancelled"}

    def _t_allocate_lot(self, a):
        b = self._planned(a["batch_id"])
        lot = self._get("lots", a["lot_id"], "lot")
        ing = lot["ingredient"]
        need = self._need(b)
        if ing not in need:
            raise _Rule("%s is not an ingredient of %s" % (self.state["ingredients"][ing]["name"],
                                                           self.state["recipes"][b["recipe"]]["name"]))
        if lot["status"] != "available":
            raise _Rule("lot %s is quarantined" % a["lot_id"])
        if lot["expires"] < b["date"]:
            raise _Rule("lot %s expires %s, before the batch date %s" % (a["lot_id"], lot["expires"], b["date"]))
        free = self._free(a["lot_id"])
        if a["grams"] > free:
            raise _Rule("lot %s has only %d g free" % (a["lot_id"], free))
        have = self._alloc_by_ing(b).get(ing, 0)
        if have + a["grams"] > need[ing]:
            raise _Rule("batch %s needs %d g of %s and already has %d g; %d g more is too much" % (
                a["batch_id"], need[ing], self.state["ingredients"][ing]["name"], have, a["grams"]))
        for k in sorted(self.state["lots"]):
            x = self.state["lots"][k]
            if k != a["lot_id"] and x["ingredient"] == ing and self._eligible(k, b) and self._free(k) > 0 and \
                    (x["expires"], k) < (lot["expires"], a["lot_id"]):
                raise _Rule("first-expiry-first-out: lot %s (expires %s, %d g free) must be used before %s" % (
                    k, x["expires"], self._free(k), a["lot_id"]))
        b["alloc"][a["lot_id"]] = b["alloc"].get(a["lot_id"], 0) + a["grams"]
        return {"ok": True, "batch_id": a["batch_id"], "lot_id": a["lot_id"], "allocated_from_lot": b["alloc"][a["lot_id"]],
                "ingredient_missing_now": need[ing] - have - a["grams"]}

    def _t_deallocate_lot(self, a):
        b = self._planned(a["batch_id"])
        self._get("lots", a["lot_id"], "lot")
        if a["lot_id"] not in b["alloc"]:
            raise _Rule("batch %s has nothing allocated from %s" % (a["batch_id"], a["lot_id"]))
        g = b["alloc"].pop(a["lot_id"])
        return {"ok": True, "batch_id": a["batch_id"], "lot_id": a["lot_id"], "grams_released": g}

    def _t_release_batch(self, a):
        b = self._planned(a["batch_id"])
        if b["date"] < self.state["today"]:
            raise _Rule("batch %s is dated in the past" % a["batch_id"])
        need, have = self._need(b), self._alloc_by_ing(b)
        short = ["%s %d/%d g" % (i, have.get(i, 0), g) for i, g in sorted(need.items()) if have.get(i, 0) != g]
        if short:
            raise _Rule("batch %s is not fully allocated: %s" % (a["batch_id"], ", ".join(short)))
        for lot, g in b["alloc"].items():
            self.state["lots"][lot]["qty"] -= g
        b["status"] = "released"
        return {"ok": True, "batch": self._batch_row(a["batch_id"])}

    def _t_quarantine_lot(self, a):
        lot = self._get("lots", a["lot_id"], "lot")
        if lot["status"] != "available":
            raise _Rule("lot %s is already quarantined" % a["lot_id"])
        users = sorted(k for k, b in self.state["batches"].items() if b["status"] == "planned" and a["lot_id"] in b["alloc"])
        if users:
            raise _Rule("lot %s is allocated to planned batch(es) %s; deallocate first" % (a["lot_id"], ", ".join(users)))
        lot["status"], lot["reason"] = "quarantined", a["reason"]
        return {"ok": True, "lot_id": a["lot_id"], "status": "quarantined"}


def _kitchen():
    ings = {}
    for iid, name, allergens in [
        ("ING-01", "Chicken thigh", []), ("ING-02", "Coconut milk", []), ("ING-03", "Green curry paste", ["crustacean"]),
        ("ING-04", "Jasmine rice", []), ("ING-05", "Peanut butter", ["peanut"]), ("ING-06", "Tahini", ["sesame"]),
        ("ING-07", "Chickpeas", []), ("ING-08", "Red lentils", []), ("ING-09", "Chopped tomatoes", []),
        ("ING-10", "Onion", []), ("ING-11", "Curry spice mix", ["mustard"]), ("ING-12", "Arborio rice", []),
        ("ING-13", "Chestnut mushrooms", []), ("ING-14", "Parmesan", ["milk"]), ("ING-15", "Cashews", ["tree_nut"]),
        ("ING-16", "Teriyaki sauce", ["soy", "gluten"]), ("ING-17", "Bell peppers", []), ("ING-18", "Tomato salsa", []),
        ("ING-19", "Flatbread", ["gluten"]), ("ING-20", "Basil pesto", ["tree_nut", "milk"]), ("ING-21", "Firm tofu", ["soy"]),
        ("ING-22", "Sesame seeds", ["sesame"]), ("ING-23", "Almond flakes", ["tree_nut"]), ("ING-24", "Penne", ["gluten"]),
        ("ING-25", "Rolled oats", ["gluten"]),
    ]:
        ings[iid] = {"name": name, "allergens": allergens}
    recipes = {}
    for rid, name, pp, status in [
        ("RC-01", "Thai Green Curry", {"ING-01": 120, "ING-02": 100, "ING-03": 20, "ING-04": 90}, "active"),
        ("RC-02", "Thai Green Curry with Tofu", {"ING-21": 120, "ING-02": 100, "ING-03": 20, "ING-04": 90}, "active"),
        ("RC-03", "Satay Chicken", {"ING-01": 120, "ING-05": 30, "ING-04": 90}, "active"),
        ("RC-04", "Satay Tofu", {"ING-21": 120, "ING-05": 30, "ING-04": 90}, "active"),
        ("RC-05", "Teriyaki Chicken", {"ING-01": 120, "ING-16": 25, "ING-04": 90, "ING-22": 3}, "active"),
        ("RC-06", "Hummus Bowl", {"ING-07": 150, "ING-06": 20, "ING-19": 80}, "active"),
        ("RC-07", "Falafel Wrap", {"ING-07": 120, "ING-06": 15, "ING-19": 100}, "active"),
        ("RC-08", "Lentil Dal", {"ING-08": 80, "ING-09": 100, "ING-10": 50, "ING-11": 5}, "active"),
        ("RC-09", "Mushroom Risotto", {"ING-12": 90, "ING-13": 80, "ING-14": 15}, "active"),
        ("RC-10", "Cashew Chicken", {"ING-01": 120, "ING-15": 25, "ING-04": 90}, "active"),
        ("RC-11", "Pesto Pasta", {"ING-20": 30, "ING-24": 100}, "active"),
        ("RC-12", "Chicken Fajita Bowl", {"ING-01": 120, "ING-17": 60, "ING-04": 90, "ING-18": 30}, "active"),
        ("RC-13", "Coconut Dal", {"ING-08": 80, "ING-02": 60, "ING-10": 40}, "active"),
        ("RC-14", "Almond Granola Pot", {"ING-23": 20, "ING-25": 60}, "active"),
        ("RC-15", "Sesame Noodles", {"ING-22": 5, "ING-06": 15, "ING-24": 100}, "active"),
        ("RC-16", "Chicken Satay Skewers", {"ING-01": 100, "ING-05": 25}, "discontinued"),
    ]:
        recipes[rid] = {"name": name, "per_portion": pp, "status": status}
    lots = {}
    for lid, ing, qty, exp, sup, status, reason in [
        ("LOT-0101", "ING-01", 4000, "2026-11-24", "Brookfield Farms", "available", ""),
        ("LOT-0102", "ING-01", 21000, "2026-11-25", "Brookfield Farms", "available", ""),
        ("LOT-0103", "ING-01", 8000, "2026-11-27", "Brookfield Farms", "available", ""),
        ("LOT-0104", "ING-01", 20000, "2026-11-30", "Tamworth Poultry", "available", ""),
        ("LOT-0201", "ING-02", 8000, "2026-12-20", "Kara Foods", "quarantined", "damaged"),
        ("LOT-0202", "ING-02", 20000, "2027-02-01", "Kara Foods", "available", ""),
        ("LOT-0301", "ING-03", 5000, "2027-01-31", "Bangkok Pantry", "available", ""),
        ("LOT-0401", "ING-04", 6000, "2026-11-26", "Riceland", "available", ""),
        ("LOT-0402", "ING-04", 9000, "2026-11-10", "Riceland", "quarantined", "expired"),
        ("LOT-0403", "ING-04", 60000, "2027-06-30", "Riceland", "available", ""),
        ("LOT-0501", "ING-05", 6000, "2027-03-01", "Nutty Co", "available", ""),
        ("LOT-0601", "ING-06", 4000, "2026-12-05", "Hanley Sesame Co", "available", ""),
        ("LOT-0602", "ING-06", 3000, "2026-12-20", "Hanley Sesame Co", "available", ""),
        ("LOT-0603", "ING-06", 1500, "2027-01-10", "Hanley Sesame Co", "available", ""),
        ("LOT-0604", "ING-06", 1200, "2026-12-12", "Levant Pantry", "available", ""),
        ("LOT-0605", "ING-06", 500, "2026-11-20", "Levant Pantry", "available", ""),
        ("LOT-0701", "ING-07", 40000, "2027-05-01", "Levant Pantry", "available", ""),
        ("LOT-0801", "ING-08", 18000, "2026-12-15", "Pulse Partners", "available", ""),
        ("LOT-0802", "ING-08", 5000, "2026-11-26", "Pulse Partners", "available", ""),
        ("LOT-0803", "ING-08", 800, "2027-01-20", "Pulse Partners", "available", ""),
        ("LOT-0804", "ING-08", 10000, "2027-02-02", "Pulse Partners", "quarantined", "damaged"),
        ("LOT-0901", "ING-09", 3000, "2026-11-22", "Sun Valley", "available", ""),
        ("LOT-0902", "ING-09", 30000, "2027-03-01", "Sun Valley", "available", ""),
        ("LOT-1001", "ING-10", 4000, "2026-11-21", "Fen Growers", "available", ""),
        ("LOT-1002", "ING-10", 2500, "2026-11-26", "Fen Growers", "available", ""),
        ("LOT-1003", "ING-10", 30000, "2026-12-02", "Fen Growers", "available", ""),
        ("LOT-1101", "ING-11", 5000, "2027-05-01", "Spice Route", "available", ""),
        ("LOT-1201", "ING-12", 18000, "2026-12-30", "Riceland", "available", ""),
        ("LOT-1202", "ING-12", 20000, "2027-02-15", "Riceland", "available", ""),
        ("LOT-1301", "ING-13", 2000, "2026-11-23", "Fen Growers", "available", ""),
        ("LOT-1302", "ING-13", 16000, "2026-11-26", "Fen Growers", "available", ""),
        ("LOT-1303", "ING-13", 12000, "2026-11-29", "Fen Growers", "available", ""),
        ("LOT-1401", "ING-14", 4000, "2027-02-10", "Dairy Hill", "available", ""),
        ("LOT-1402", "ING-14", 6000, "2027-04-10", "Dairy Hill", "available", ""),
        ("LOT-1501", "ING-15", 5000, "2027-04-01", "Nutty Co", "available", ""),
        ("LOT-1601", "ING-16", 9000, "2027-06-01", "Kara Foods", "available", ""),
        ("LOT-1701", "ING-17", 12000, "2026-11-27", "Sun Valley", "available", ""),
        ("LOT-1801", "ING-18", 3000, "2026-12-01", "Sun Valley", "available", ""),
        ("LOT-1802", "ING-18", 5000, "2026-12-10", "Sun Valley", "available", ""),
        ("LOT-1901", "ING-19", 30000, "2026-11-28", "Oven Fresh", "available", ""),
        ("LOT-2001", "ING-20", 4000, "2026-12-08", "Kara Foods", "available", ""),
        ("LOT-2101", "ING-21", 20000, "2026-12-03", "Kara Foods", "available", ""),
        ("LOT-2201", "ING-22", 2000, "2026-12-15", "Hanley Sesame Co", "available", ""),
        ("LOT-2301", "ING-23", 3000, "2027-03-03", "Nutty Co", "available", ""),
        ("LOT-2401", "ING-24", 40000, "2027-08-01", "Oven Fresh", "available", ""),
        ("LOT-2501", "ING-25", 20000, "2027-05-05", "Oven Fresh", "available", ""),
    ]:
        lots[lid] = {"ingredient": ing, "qty": qty, "expires": exp, "supplier": sup, "status": status, "reason": reason}
    batches = {}
    for bid, rid, portions, date, line, status, alloc in [
        ("PB-850", "RC-03", 150, "11-23", "L1", "released", {}),
        ("PB-853", "RC-03", 60, "11-23", "L1", "planned", {}),
        ("PB-854", "RC-09", 150, "11-23", "L1", "planned", {}),
        ("PB-852", "RC-04", 90, "11-23", "L2", "planned", {}),
        ("PB-820", "RC-01", 200, "11-24", "L1", "planned", {}),
        ("PB-821", "RC-12", 100, "11-24", "L1", "planned", {"LOT-0401": 1000}),
        ("PB-822", "RC-08", 150, "11-24", "L2", "planned", {}),
        ("PB-810", "RC-03", 100, "11-24", "L3", "planned", {}),
        ("PB-813", "RC-14", 60, "11-24", "L3", "released", {}),
        ("PB-814", "RC-15", 70, "11-24", "L3", "planned", {}),
        ("PB-830", "RC-09", 120, "11-25", "L1", "planned", {"LOT-1201": 10800, "LOT-1302": 9600, "LOT-1401": 1800}),
        ("PB-831", "RC-08", 250, "11-25", "L1", "planned", {}),
        ("PB-832", "RC-09", 80, "11-25", "L2", "planned", {"LOT-1201": 7200, "LOT-1302": 6400, "LOT-1401": 1200}),
        ("PB-833", "RC-06", 100, "11-25", "L2", "planned", {"LOT-0701": 15000, "LOT-0601": 2000}),
        ("PB-811", "RC-11", 90, "11-25", "L3", "planned", {}),
        ("PB-817", "RC-04", 80, "11-25", "L3", "cancelled", {}),
        ("PB-801", "RC-09", 180, "11-26", "L1", "planned", {}),
        ("PB-802", "RC-05", 120, "11-26", "L1", "planned", {"LOT-0103": 2000}),
        ("PB-851", "RC-03", 120, "11-26", "L2", "planned", {}),
        ("PB-805", "RC-12", 80, "11-26", "L2", "planned", {}),
        ("PB-804", "RC-13", 80, "11-26", "L3", "planned", {}),
        ("PB-840", "RC-01", 250, "11-27", "L1", "planned", {}),
        ("PB-841", "RC-05", 120, "11-27", "L1", "planned", {}),
        ("PB-842", "RC-08", 200, "11-27", "L2", "planned",
         {"LOT-0801": 16000, "LOT-0902": 20000, "LOT-1003": 10000, "LOT-1101": 1000}),
        ("PB-843", "RC-07", 60, "11-27", "L2", "planned", {"LOT-0602": 900}),
        ("PB-812", "RC-10", 120, "11-27", "L3", "planned", {}),
        ("PB-844", "RC-13", 100, "11-27", "L3", "planned", {"LOT-1001": 4000}),
        ("PB-816", "RC-03", 140, "11-19", "L3", "released", {}),
        ("PB-818", "RC-08", 200, "11-20", "L2", "released", {}),
    ]:
        batches[bid] = {"recipe": rid, "portions": portions, "date": "2026-" + date, "line": line, "status": status,
                        "alloc": alloc}
    st = {"today": "2026-11-23",
          "lines": {"L1": {"name": "Line 1", "capacity": 400, "nut_free": False},
                    "L2": {"name": "Line 2", "capacity": 300, "nut_free": False},
                    "L3": {"name": "Line 3", "capacity": 250, "nut_free": True}},
          "ingredients": ings, "recipes": recipes, "lots": lots, "batches": batches, "next_batch": 900}
    env = KitchenEnv(st)
    for k in lots:
        assert env._free(k) >= 0, k
    for k, b in batches.items():
        if b["status"] in ("planned", "released"):
            assert env._used(b["date"], b["line"]) <= st["lines"][b["line"]]["capacity"], k
        for lot, g in b["alloc"].items():
            assert lots[lot]["ingredient"] in recipes[b["recipe"]]["per_portion"], (k, lot)
    return st


def _alloc_ok(i, recipe, portions, date):
    """Predicate for an allocation dict: exactly the recipe's needs, only from usable lots."""
    need = dict((ing, g * portions) for ing, g in i["recipes"][recipe]["per_portion"].items())

    def ok(alloc):
        if not isinstance(alloc, dict):
            return False
        have = {}
        for lot, g in alloc.items():
            x = i["lots"].get(lot)
            if x is None or x["status"] != "available" or x["expires"] < date or not isinstance(g, int) or g <= 0:
                return False
            have[x["ingredient"]] = have.get(x["ingredient"], 0) + g
        return have == need
    return ok


def _qty_after(i, alloc):
    return dict((lot, {"qty": i["lots"][lot]["qty"] - g}) for lot, g in alloc.items())


_T17_ALLOC = {"LOT-0103": 6000, "LOT-0104": 12000, "LOT-0202": 15000, "LOT-0301": 3000, "LOT-0401": 5000,
              "LOT-0403": 8500}

_add("T17", "kitchen", _kitchen(),
     "Please plan 150 portions of Thai Green Curry (the chicken one, not the tofu version) for Thursday 26 "
     "November 2026, on a production line that can take the whole batch that day. Allocate every ingredient "
     "following the kitchen's stock rules, and once it is fully allocated, release the batch to production so the "
     "team can start prepping. Don't move, resize or re-allocate any other batch, and nothing else in the plan or "
     "in the stock should change.",
     [["get_kitchen_info", {}], ["search_recipes", {"query": "Thai Green Curry"}], ["get_recipe", {"recipe_id": "RC-01"}],
      ["list_batches", {"date": "2026-11-26"}],
      ["create_batch", {"recipe_id": "RC-01", "portions": 150, "date": "2026-11-26", "line": "L3"}],
      ["list_lots", {"ingredient_id": "ING-01"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0103", "grams": 6000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0104", "grams": 12000}],
      ["list_lots", {"ingredient_id": "ING-02"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0202", "grams": 15000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0301", "grams": 3000}],
      ["list_lots", {"ingredient_id": "ING-04"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0401", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0403", "grams": 8500}],
      ["release_batch", {"batch_id": "PB-900"}]],
     [["create_batch", {"recipe_id": "RC-01", "portions": 150, "date": "2026-11-26", "line": "L3"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0104", "grams": 18000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0202", "grams": 15000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0301", "grams": 3000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0403", "grams": 13500}]],
     lambda s, i: _final(s, i, changes={"lots": _qty_after(i, _T17_ALLOC)}, new={"batches": [
         {"recipe": "RC-01", "portions": 150, "date": "2026-11-26", "line": "L3", "status": "released",
          "alloc": _T17_ALLOC}]}),
     mentions=["Thai Green Curry"], seq=[[4, 6, 7, 9, 10, 12, 13, 14]])

_add("T18", "kitchen", _kitchen(),
     "Hanley Sesame Co has recalled its tahini lots that expire in December 2026. Quarantine those lots with reason "
     "recall. Every planned batch that had tahini allocated from them needs replacement tahini from our other usable "
     "lots, following the stock rules; deal with the affected batches in date order. If there isn't enough usable "
     "tahini left for a batch, reduce that batch's portions to the largest number the remaining tahini can fully "
     "cover and allocate for that. Hanley's other tahini and their other products are not affected, and nothing "
     "else should change.",
     [["find_ingredients", {"query": "tahini"}], ["list_lots", {"ingredient_id": "ING-06"}],
      ["get_batch", {"batch_id": "PB-833"}], ["get_batch", {"batch_id": "PB-843"}],
      ["deallocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0601"}],
      ["deallocate_lot", {"batch_id": "PB-843", "lot_id": "LOT-0602"}],
      ["quarantine_lot", {"lot_id": "LOT-0601", "reason": "recall"}],
      ["quarantine_lot", {"lot_id": "LOT-0602", "reason": "recall"}],
      ["allocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0604", "grams": 1200}],
      ["allocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0603", "grams": 800}],
      ["update_batch", {"batch_id": "PB-843", "portions": 46}],
      ["allocate_lot", {"batch_id": "PB-843", "lot_id": "LOT-0603", "grams": 690}]],
     [["deallocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0601"}],
      ["deallocate_lot", {"batch_id": "PB-843", "lot_id": "LOT-0602"}],
      ["quarantine_lot", {"lot_id": "LOT-0601", "reason": "recall"}],
      ["quarantine_lot", {"lot_id": "LOT-0602", "reason": "recall"}],
      ["quarantine_lot", {"lot_id": "LOT-0603", "reason": "recall"}],
      ["allocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0604", "grams": 1200}],
      ["update_batch", {"batch_id": "PB-833", "portions": 60}]],
     lambda s, i: _final(s, i, changes={
         "lots": {"LOT-0601": {"status": "quarantined", "reason": "recall"},
                  "LOT-0602": {"status": "quarantined", "reason": "recall"}},
         "batches": {"PB-833": {"alloc": {"LOT-0701": 15000, "LOT-0604": 1200, "LOT-0603": 800}},
                     "PB-843": {"portions": 46, "alloc": {"LOT-0603": 690}}}}),
     mentions=["Hanley Sesame Co"], seq=[[8, 9, 10, 11]])

_add("T19", "kitchen", _kitchen(),
     "Line 3 was converted to nut-free last week, but some batches containing nuts were planned on it before the "
     "change. For every batch on Line 3 from today onwards that is still planned and whose recipe contains peanut or "
     "tree nut, move it to Line 1 if Line 1 has room for it that day, otherwise to Line 2 if Line 2 has room; if "
     "neither can take it, leave it where it is and I'll sort it out by hand. Recipes without nuts stay on Line 3, "
     "released batches can't be touched, and nothing else should change.",
     [["get_kitchen_info", {}], ["list_batches", {"line": "L3"}], ["list_batches", {"line": "L3", "page": 2}],
      ["get_recipe", {"recipe_id": "RC-03"}], ["get_recipe", {"recipe_id": "RC-15"}],
      ["get_recipe", {"recipe_id": "RC-11"}], ["get_recipe", {"recipe_id": "RC-13"}],
      ["get_recipe", {"recipe_id": "RC-10"}],
      ["list_batches", {"date": "2026-11-24"}], ["update_batch", {"batch_id": "PB-810", "line": "L1"}],
      ["list_batches", {"date": "2026-11-25"}], ["update_batch", {"batch_id": "PB-811", "line": "L2"}],
      ["list_batches", {"date": "2026-11-27"}]],
     [["update_batch", {"batch_id": "PB-810", "line": "L1"}], ["update_batch", {"batch_id": "PB-814", "line": "L2"}],
      ["update_batch", {"batch_id": "PB-811", "line": "L2"}]],
     lambda s, i: _final(s, i, changes={"batches": {"PB-810": {"line": "L1"}, "PB-811": {"line": "L2"}}}),
     mentions=["Line 3", "Line 1", "Line 2"])

_add("T20", "kitchen", _kitchen(),
     "Can we make more of Friday's Lentil Dal (the batch planned for 27 November)? Raise it from 200 to 260 portions "
     "if both the line capacity and the usable stock allow it. If they don't, raise it to the largest number of "
     "portions that both allow, keeping it on its current line - don't move or shrink anything else to make room. "
     "Then allocate whatever is still missing, following the stock rules, and release the batch. Nothing else "
     "should change.",
     [["get_kitchen_info", {}], ["list_batches", {"date": "2026-11-27"}], ["get_batch", {"batch_id": "PB-842"}],
      ["list_lots", {"ingredient_id": "ING-08"}], ["list_lots", {"ingredient_id": "ING-09"}],
      ["list_lots", {"ingredient_id": "ING-10"}], ["list_lots", {"ingredient_id": "ING-11"}],
      ["update_batch", {"batch_id": "PB-842", "portions": 235}],
      ["allocate_lot", {"batch_id": "PB-842", "lot_id": "LOT-0801", "grams": 2000}],
      ["allocate_lot", {"batch_id": "PB-842", "lot_id": "LOT-0803", "grams": 800}],
      ["allocate_lot", {"batch_id": "PB-842", "lot_id": "LOT-0902", "grams": 3500}],
      ["allocate_lot", {"batch_id": "PB-842", "lot_id": "LOT-1003", "grams": 1750}],
      ["allocate_lot", {"batch_id": "PB-842", "lot_id": "LOT-1101", "grams": 175}],
      ["release_batch", {"batch_id": "PB-842"}]],
     [["update_batch", {"batch_id": "PB-842", "portions": 240}],
      ["allocate_lot", {"batch_id": "PB-842", "lot_id": "LOT-0801", "grams": 2000}],
      ["allocate_lot", {"batch_id": "PB-842", "lot_id": "LOT-0803", "grams": 800}],
      ["allocate_lot", {"batch_id": "PB-842", "lot_id": "LOT-0902", "grams": 4000}],
      ["allocate_lot", {"batch_id": "PB-842", "lot_id": "LOT-1003", "grams": 2000}],
      ["allocate_lot", {"batch_id": "PB-842", "lot_id": "LOT-1101", "grams": 200}]],
     lambda s, i: _final(s, i, changes={
         "batches": {"PB-842": {"portions": 235, "status": "released", "alloc": {
             "LOT-0801": 18000, "LOT-0803": 800, "LOT-0902": 23500, "LOT-1003": 11750, "LOT-1101": 1175}}},
         "lots": _qty_after(i, {"LOT-0801": 18000, "LOT-0803": 800, "LOT-0902": 23500, "LOT-1003": 11750,
                                "LOT-1101": 1175})}),
     mentions=["Lentil Dal"], seq=[[7, 8, 9, 10, 11, 12, 13]])

_add("T21", "kitchen", _kitchen(),
     "Time for the expiry sweep. Every lot whose expiry date has already passed must be quarantined with reason "
     "expired, whether or not it still has stock in it. If an expired lot is still allocated to a planned batch, "
     "take that allocation off first and replace it from usable stock following the stock rules, so the batch is "
     "no worse off. Lots that expire today or later are fine, lots that are already quarantined stay as they are, "
     "and nothing else should change.",
     [["get_kitchen_info", {}], ["list_lots", {"expires_before": "2026-11-23"}],
      ["quarantine_lot", {"lot_id": "LOT-0605", "reason": "expired"}],
      ["quarantine_lot", {"lot_id": "LOT-0901", "reason": "expired"}],
      ["quarantine_lot", {"lot_id": "LOT-1001", "reason": "expired"}],
      ["get_batch", {"batch_id": "PB-844"}],
      ["deallocate_lot", {"batch_id": "PB-844", "lot_id": "LOT-1001"}],
      ["quarantine_lot", {"lot_id": "LOT-1001", "reason": "expired"}],
      ["list_lots", {"ingredient_id": "ING-10"}],
      ["allocate_lot", {"batch_id": "PB-844", "lot_id": "LOT-1002", "grams": 4000}],
      ["allocate_lot", {"batch_id": "PB-844", "lot_id": "LOT-1003", "grams": 4000}]],
     [["quarantine_lot", {"lot_id": "LOT-0605", "reason": "expired"}],
      ["quarantine_lot", {"lot_id": "LOT-0901", "reason": "expired"}],
      ["quarantine_lot", {"lot_id": "LOT-1301", "reason": "expired"}],
      ["deallocate_lot", {"batch_id": "PB-844", "lot_id": "LOT-1001"}],
      ["quarantine_lot", {"lot_id": "LOT-1001", "reason": "expired"}],
      ["allocate_lot", {"batch_id": "PB-844", "lot_id": "LOT-1003", "grams": 4000}]],
     lambda s, i: _final(s, i, changes={
         "lots": {"LOT-0605": {"status": "quarantined", "reason": "expired"},
                  "LOT-0901": {"status": "quarantined", "reason": "expired"},
                  "LOT-1001": {"status": "quarantined", "reason": "expired"}},
         "batches": {"PB-844": {"alloc": {"LOT-1003": 4000}}}}),
     ref_errors=[4, 9])

_add("T22", "kitchen", _kitchen(),
     "We are discontinuing Satay Chicken. Cancel every Satay Chicken batch from today onwards that is still only "
     "planned - anything already released goes ahead as it is. For each batch you cancel, plan a Teriyaki Chicken "
     "batch in its place: same date, same line, same number of portions, with nothing allocated yet. Satay Tofu "
     "stays on the menu, so leave its batches alone, and don't change anything else.",
     [["search_recipes", {"query": "Satay"}], ["search_recipes", {"query": "Teriyaki"}],
      ["list_batches", {"recipe_id": "RC-03"}],
      ["create_batch", {"recipe_id": "RC-05", "portions": 60, "date": "2026-11-23", "line": "L1"}],
      ["cancel_batch", {"batch_id": "PB-853"}],
      ["create_batch", {"recipe_id": "RC-05", "portions": 60, "date": "2026-11-23", "line": "L1"}],
      ["cancel_batch", {"batch_id": "PB-810"}],
      ["create_batch", {"recipe_id": "RC-05", "portions": 100, "date": "2026-11-24", "line": "L3"}],
      ["cancel_batch", {"batch_id": "PB-851"}],
      ["create_batch", {"recipe_id": "RC-05", "portions": 120, "date": "2026-11-26", "line": "L2"}]],
     [["cancel_batch", {"batch_id": "PB-853"}], ["cancel_batch", {"batch_id": "PB-810"}],
      ["cancel_batch", {"batch_id": "PB-851"}], ["cancel_batch", {"batch_id": "PB-852"}],
      ["create_batch", {"recipe_id": "RC-05", "portions": 60, "date": "2026-11-23", "line": "L1"}],
      ["create_batch", {"recipe_id": "RC-05", "portions": 100, "date": "2026-11-24", "line": "L3"}],
      ["create_batch", {"recipe_id": "RC-05", "portions": 120, "date": "2026-11-26", "line": "L2"}],
      ["create_batch", {"recipe_id": "RC-05", "portions": 90, "date": "2026-11-23", "line": "L2"}]],
     lambda s, i: _final(s, i, changes={"batches": dict(
         (b, {"status": "cancelled"}) for b in ("PB-853", "PB-810", "PB-851"))},
         new={"batches": [
             {"recipe": "RC-05", "portions": 60, "date": "2026-11-23", "line": "L1", "status": "planned", "alloc": {}},
             {"recipe": "RC-05", "portions": 100, "date": "2026-11-24", "line": "L3", "status": "planned", "alloc": {}},
             {"recipe": "RC-05", "portions": 120, "date": "2026-11-26", "line": "L2", "status": "planned", "alloc": {}}]}),
     ref_errors=[3], mentions=["Satay Chicken", "Teriyaki Chicken", "Satay Tofu"])


def _c23(s, i):
    try:
        alloc = s["batches"]["PB-832"]["alloc"]
        if not _alloc_ok(i, "RC-09", 200, "2026-11-25")(alloc):
            return False
    except (KeyError, TypeError):
        return False
    return _final(s, i, changes={
        "batches": {"PB-830": {"status": "cancelled", "alloc": {}},
                    "PB-832": {"portions": 200, "status": "released", "alloc": alloc}},
        "lots": _qty_after(i, alloc)})


_add("T23", "kitchen", _kitchen(),
     "There are two Mushroom Risotto batches planned for Wednesday 25 November. Merge them into a single batch with "
     "the combined number of portions: keep the Line 1 batch and grow it if Line 1 has room for the combined amount "
     "that day; otherwise keep the other batch, grow that one, and cancel the Line 1 batch. Then make sure the "
     "merged batch is fully allocated following the stock rules, and release it to production. No other batch "
     "should be moved or changed.",
     [["get_kitchen_info", {}], ["list_batches", {"date": "2026-11-25"}],
      ["get_batch", {"batch_id": "PB-830"}], ["get_batch", {"batch_id": "PB-832"}],
      ["update_batch", {"batch_id": "PB-830", "portions": 200}],
      ["cancel_batch", {"batch_id": "PB-830"}], ["update_batch", {"batch_id": "PB-832", "portions": 200}],
      ["list_lots", {"ingredient_id": "ING-12"}], ["list_lots", {"ingredient_id": "ING-13"}],
      ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1201", "grams": 10800}],
      ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1302", "grams": 9600}],
      ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1401", "grams": 1800}],
      ["release_batch", {"batch_id": "PB-832"}]],
     [["update_batch", {"batch_id": "PB-832", "portions": 200}],
      ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1202", "grams": 10800}],
      ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1303", "grams": 9600}],
      ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1401", "grams": 1000}],
      ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1402", "grams": 800}],
      ["release_batch", {"batch_id": "PB-832"}]],
     _c23, ref_errors=[4], mentions=["Mushroom Risotto", "Line 1"])

_T24_ALLOC = {"LOT-0102": 19200, "LOT-1701": 9600, "LOT-0401": 5000, "LOT-0403": 9400, "LOT-1801": 3000,
              "LOT-1802": 1800}

_add("T24", "kitchen", _kitchen(),
     "Chicken thigh lot LOT-0102 expires on Wednesday and I don't want any of it wasted. Plan a new Chicken Fajita "
     "Bowl batch for Wednesday 25 November that uses as much of that lot as possible: as many portions as the lot's "
     "free stock can cover, but no more than the line can take - put it on whichever line has the most room that "
     "day for this recipe. Allocate all of its ingredients following the stock rules, but don't release it yet. "
     "Nothing else should change.",
     [["get_kitchen_info", {}], ["list_lots", {"ingredient_id": "ING-01"}],
      ["search_recipes", {"query": "Fajita"}], ["get_recipe", {"recipe_id": "RC-12"}],
      ["list_batches", {"date": "2026-11-25"}],
      ["create_batch", {"recipe_id": "RC-12", "portions": 160, "date": "2026-11-25", "line": "L3"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0102", "grams": 19200}],
      ["list_lots", {"ingredient_id": "ING-17"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1701", "grams": 9600}],
      ["list_lots", {"ingredient_id": "ING-04"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0401", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0403", "grams": 9400}],
      ["list_lots", {"ingredient_id": "ING-18"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1801", "grams": 3000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1802", "grams": 1800}]],
     [["create_batch", {"recipe_id": "RC-12", "portions": 120, "date": "2026-11-25", "line": "L2"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0102", "grams": 14400}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1701", "grams": 7200}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0401", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0403", "grams": 5800}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1801", "grams": 3000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1802", "grams": 600}]],
     lambda s, i: _final(s, i, new={"batches": [
         {"recipe": "RC-12", "portions": 160, "date": "2026-11-25", "line": "L3", "status": "planned",
          "alloc": _T24_ALLOC}]}),
     mentions=["Chicken Fajita Bowl"], seq=[[5, 6, 8, 10, 11, 13, 14]])


# --------------------------------------------------------------------------
# 4. content moderation: posts, reports, moderators, strikes and suspensions
# --------------------------------------------------------------------------
_REASONS = ["spam", "harassment", "hate", "misinfo", "threat", "self_harm"]
_HIGH_TIER_REASONS = ("threat", "self_harm")


class ModerationEnv(_Env):
    TOOLS = [
        _fn("get_mod_policy", "Today's date, communities and their languages, and the moderation policy that the "
            "write tools enforce."),
        _fn("list_reports", "Reports, filtered, oldest first; 8 per page.",
            {"status": _s("Only this status.", ["open", "escalated", "actioned", "dismissed"]),
             "reason": _s("Only this reason.", _REASONS), "community": _s("Only this community, e.g. c/gardening."),
             "assignee": _s("Only reports assigned to this moderator id."),
             "reporter_id": _s("Only reports filed by this user id."), "page": _PAGE}),
        _fn("list_posts", "Posts, filtered, oldest first; 8 per page, with a count of their reports by status.",
            {"community": _s("Only this community."), "status": _s("Only this status.", ["visible", "hidden", "removed"]),
             "author_id": _s("Only posts by this user id."), "page": _PAGE}),
        _fn("get_post", "One post with every report filed on it.", {"post_id": _s("Post id, e.g. PO-2001.")}, ["post_id"]),
        _fn("find_users", "Search users by handle (case-insensitive substring; a leading @ is ignored).",
            {"query": _s("Text that must appear in the handle.")}, ["query"]),
        _fn("get_user", "One user: status, suspension end, strikes (and the posts they were given for), verified "
            "flag, join date, trusted-flagger flag.", {"user_id": _s("User id, e.g. U-101.")}, ["user_id"]),
        _fn("list_moderators", "All moderators: tier, languages, active flag, number of open or escalated reports "
            "they hold, and their maximum."),
        _fn("assign_report", "Assign (or re-assign) an open or escalated report to a moderator.",
            {"report_id": _s("Report id, e.g. RP-3001."), "moderator_id": _s("Moderator id, e.g. M-01.")},
            ["report_id", "moderator_id"]),
        _fn("resolve_report", "Close an open or escalated report.",
            {"report_id": _s("Report id."), "outcome": _s("Outcome.", ["actioned", "dismissed"])}, ["report_id", "outcome"]),
        _fn("escalate_report", "Escalate an open report to a tier 2 or tier 3 moderator.",
            {"report_id": _s("Report id."), "moderator_id": _s("Moderator id.")}, ["report_id", "moderator_id"]),
        _fn("set_post_status", "Make a post visible, hide it, or remove it.",
            {"post_id": _s("Post id."), "status": _s("New status.", ["visible", "hidden", "removed"])}, ["post_id", "status"]),
        _fn("add_strike", "Give a user a strike for one of their removed posts.",
            {"user_id": _s("User id."), "post_id": _s("The removed post the strike is for.")}, ["user_id", "post_id"]),
        _fn("remove_strike", "Take back the strike a user was given for a post.",
            {"user_id": _s("User id."), "post_id": _s("Post id.")}, ["user_id", "post_id"]),
        _fn("set_user_status", "Suspend, ban or reinstate a user. Every change needs the id of the moderator who "
            "signs it off.",
            {"user_id": _s("User id."), "status": _s("New status.", ["active", "suspended", "banned"]),
             "approved_by": _s("Moderator id signing off.")}, ["user_id", "status", "approved_by"]),
    ]

    def _load(self, mid):
        return len([1 for r in self.state["reports"].values() if r["assignee"] == mid and r["status"] in ("open", "escalated")])

    def _lang(self, rid):
        return self.state["communities"][self.state["posts"][self.state["reports"][rid]["post"]]["community"]]

    def _needs_high(self, rid):
        r = self.state["reports"][rid]
        author = self.state["users"][self.state["posts"][r["post"]]["author"]]
        return r["reason"] in _HIGH_TIER_REASONS or author["verified"] or r["status"] == "escalated"

    def _can_take(self, rid, mid, min_tier=1):
        m = self._get("moderators", mid, "moderator")
        if not m["active"]:
            raise _Rule("moderator %s is not active" % mid)
        lang = self._lang(rid)
        if lang not in m["languages"]:
            raise _Rule("report %s is in a '%s' community; %s does not moderate in that language" % (rid, lang, mid))
        if self._needs_high(rid):
            min_tier = max(min_tier, 2)
        if m["tier"] < min_tier:
            raise _Rule("report %s needs a tier %d+ moderator; %s is tier %d" % (rid, min_tier, mid, m["tier"]))
        if self.state["reports"][rid]["assignee"] == mid:
            raise _Rule("report %s is already assigned to %s" % (rid, mid))
        if self._load(mid) >= m["max_open"]:
            raise _Rule("moderator %s already holds %d open reports (maximum %d)" % (mid, self._load(mid), m["max_open"]))

    def _rep_row(self, k):
        r = self.state["reports"][k]
        p = self.state["posts"][r["post"]]
        return {"report_id": k, "post_id": r["post"], "community": p["community"], "post_author_id": p["author"],
                "post_author": self.state["users"][p["author"]]["handle"],
                "post_author_joined": self.state["users"][p["author"]]["joined"], "reporter_id": r["reporter"],
                "reason": r["reason"], "created": r["created"], "status": r["status"], "assignee": r["assignee"]}

    def _post_row(self, k):
        p = self.state["posts"][k]
        counts = {}
        for r in self.state["reports"].values():
            if r["post"] == k:
                counts[r["status"]] = counts.get(r["status"], 0) + 1
        return {"post_id": k, "author_id": p["author"], "author": self.state["users"][p["author"]]["handle"],
                "community": p["community"], "created": p["created"], "status": p["status"], "title": p["title"],
                "reports_by_status": counts}

    def _t_get_mod_policy(self, a):
        p = self.state["policy"]
        return {"today": self.state["today"], "communities": self.state["communities"], "policy": {
            "suspension_at_strikes": p["suspend_at"], "ban_at_strikes": p["ban_at"], "suspension_days": p["suspend_days"],
            "trusted_flagger_new_account_days": p["new_account_days"],
            "rules": [
                "A report must be handled by an active moderator who moderates in the language of the post's community.",
                "Reports with reason threat or self_harm, reports about posts by verified users, and escalated "
                "reports may only be held by tier 2 or tier 3 moderators.",
                "A moderator may hold at most max_open reports that are open or escalated.",
                "self_harm reports may never be dismissed; they are escalated or actioned.",
                "A user gets at most one strike per removed post, and only for their own removed posts.",
                "A user who reaches suspension_at_strikes strikes must be suspended for suspension_days days; a user "
                "who reaches ban_at_strikes must be banned. Suspensions need sign-off from a tier 2+ moderator, bans "
                "from a tier 3 moderator. A suspended user may be reinstated early only once they are below "
                "suspension_at_strikes strikes.",
                "A trusted flagger's spam report may be actioned without review when the post's author account is "
                "less than trusted_flagger_new_account_days days old (joined after today minus that many days).",
            ]}}

    def _t_list_reports(self, a):
        rows = [k for _, k in sorted((r["created"], k) for k, r in self.state["reports"].items()
                                     if ("status" not in a or r["status"] == a["status"])
                                     and ("reason" not in a or r["reason"] == a["reason"])
                                     and ("community" not in a or self.state["posts"][r["post"]]["community"] == a["community"])
                                     and ("assignee" not in a or r["assignee"] == a["assignee"])
                                     and ("reporter_id" not in a or r["reporter"] == a["reporter_id"]))]
        meta, rows = _paged(rows, a.get("page", 1), 8)
        meta["reports"] = [self._rep_row(k) for k in rows]
        return meta

    def _t_list_posts(self, a):
        rows = [k for _, k in sorted((p["created"], k) for k, p in self.state["posts"].items()
                                     if ("community" not in a or p["community"] == a["community"])
                                     and ("status" not in a or p["status"] == a["status"])
                                     and ("author_id" not in a or p["author"] == a["author_id"]))]
        meta, rows = _paged(rows, a.get("page", 1), 8)
        meta["posts"] = [self._post_row(k) for k in rows]
        return meta

    def _t_get_post(self, a):
        self._get("posts", a["post_id"], "post")
        row = self._post_row(a["post_id"])
        row["reports"] = [self._rep_row(k) for _, k in sorted((r["created"], k) for k, r in self.state["reports"].items()
                                                               if r["post"] == a["post_id"])]
        return row

    def _t_find_users(self, a):
        q = a["query"].strip().lower().lstrip("@")
        if not q:
            raise _Rule("query must not be empty")
        return {"users": [{"user_id": k, "handle": u["handle"], "status": u["status"], "verified": u["verified"]}
                          for k, u in sorted(self.state["users"].items()) if q in u["handle"].lower()]}

    def _t_get_user(self, a):
        u = self._get("users", a["user_id"], "user")
        return {"user_id": a["user_id"], "handle": u["handle"], "status": u["status"],
                "suspended_until": u["suspended_until"], "strikes": len(u["strike_posts"]),
                "strike_posts": list(u["strike_posts"]), "verified": u["verified"], "joined": u["joined"],
                "trusted_flagger": u["trusted_flagger"]}

    def _t_list_moderators(self, a):
        return {"moderators": [{"moderator_id": k, "name": m["name"], "tier": m["tier"], "languages": m["languages"],
                                "active": m["active"], "open_reports": self._load(k), "max_open": m["max_open"]}
                               for k, m in sorted(self.state["moderators"].items())]}

    def _t_assign_report(self, a):
        r = self._get("reports", a["report_id"], "report")
        if r["status"] not in ("open", "escalated"):
            raise _Rule("report %s is %s" % (a["report_id"], r["status"]))
        self._can_take(a["report_id"], a["moderator_id"])
        r["assignee"] = a["moderator_id"]
        return {"ok": True, "report": self._rep_row(a["report_id"])}

    def _t_resolve_report(self, a):
        r = self._get("reports", a["report_id"], "report")
        if r["status"] not in ("open", "escalated"):
            raise _Rule("report %s is already %s" % (a["report_id"], r["status"]))
        if a["outcome"] == "dismissed" and r["reason"] == "self_harm":
            raise _Rule("self_harm reports may never be dismissed")
        r["status"] = a["outcome"]
        return {"ok": True, "report": self._rep_row(a["report_id"])}

    def _t_escalate_report(self, a):
        r = self._get("reports", a["report_id"], "report")
        if r["status"] != "open":
            raise _Rule("only open reports can be escalated; %s is %s" % (a["report_id"], r["status"]))
        self._can_take(a["report_id"], a["moderator_id"], 2)
        r["status"], r["assignee"] = "escalated", a["moderator_id"]
        return {"ok": True, "report": self._rep_row(a["report_id"])}

    def _t_set_post_status(self, a):
        p = self._get("posts", a["post_id"], "post")
        if p["status"] == a["status"]:
            raise _Rule("post %s is already %s" % (a["post_id"], p["status"]))
        p["status"] = a["status"]
        return {"ok": True, "post": self._post_row(a["post_id"])}

    def _t_add_strike(self, a):
        u = self._get("users", a["user_id"], "user")
        p = self._get("posts", a["post_id"], "post")
        if p["author"] != a["user_id"]:
            raise _Rule("post %s was written by %s, not %s" % (a["post_id"], p["author"], a["user_id"]))
        if p["status"] != "removed":
            raise _Rule("post %s is %s; strikes are only given for removed posts" % (a["post_id"], p["status"]))
        if a["post_id"] in u["strike_posts"]:
            raise _Rule("%s already has a strike for %s" % (a["user_id"], a["post_id"]))
        u["strike_posts"].append(a["post_id"])
        return {"ok": True, "user_id": a["user_id"], "strikes": len(u["strike_posts"])}

    def _t_remove_strike(self, a):
        u = self._get("users", a["user_id"], "user")
        if a["post_id"] not in u["strike_posts"]:
            raise _Rule("%s has no strike for %s" % (a["user_id"], a["post_id"][:30]))
        u["strike_posts"].remove(a["post_id"])
        return {"ok": True, "user_id": a["user_id"], "strikes": len(u["strike_posts"])}

    def _t_set_user_status(self, a):
        u = self._get("users", a["user_id"], "user")
        m = self._get("moderators", a["approved_by"], "moderator")
        p = self.state["policy"]
        n = len(u["strike_posts"])
        if not m["active"]:
            raise _Rule("moderator %s is not active" % a["approved_by"])
        if u["status"] == a["status"]:
            raise _Rule("user %s is already %s" % (a["user_id"], u["status"]))
        if u["status"] == "banned":
            raise _Rule("user %s is banned; bans cannot be changed here" % a["user_id"])
        if a["status"] == "banned":
            if m["tier"] < 3:
                raise _Rule("bans need sign-off from a tier 3 moderator; %s is tier %d" % (a["approved_by"], m["tier"]))
            if n < p["ban_at"]:
                raise _Rule("user %s has %d strikes; a ban needs %d" % (a["user_id"], n, p["ban_at"]))
            u["status"], u["suspended_until"] = "banned", ""
        elif a["status"] == "suspended":
            if m["tier"] < 2:
                raise _Rule("suspensions need sign-off from a tier 2+ moderator; %s is tier %d" % (a["approved_by"], m["tier"]))
            if n < p["suspend_at"]:
                raise _Rule("user %s has %d strikes; a suspension needs %d" % (a["user_id"], n, p["suspend_at"]))
            u["status"], u["suspended_until"] = "suspended", _plus_days(self.state["today"], p["suspend_days"])
        else:
            if m["tier"] < 2:
                raise _Rule("reinstatements need sign-off from a tier 2+ moderator")
            if n >= p["suspend_at"]:
                raise _Rule("user %s still has %d strikes and cannot be reinstated early" % (a["user_id"], n))
            u["status"], u["suspended_until"] = "active", ""
        return {"ok": True, "user_id": a["user_id"], "status": u["status"], "suspended_until": u["suspended_until"]}


def _mod():
    users = {}
    for uid, handle, joined, strikes, status, until, verified, trusted in [
        ("U-101", "pixel_pete", "2023-04-01", ["PO-1901", "PO-1902"], "active", "", False, False),
        ("U-102", "arcade_annie", "2021-02-11", [], "active", "", True, False),
        ("U-103", "joystick_jay", "2022-08-19", [], "active", "", False, False),
        ("U-104", "bitmap_bea", "2024-01-30", [], "active", "", False, False),
        ("U-105", "harbour_watch", "2020-05-05", [], "active", "", True, False),
        ("U-110", "marigold_jo", "2022-03-03", ["PO-2010", "PO-2011", "PO-2012"], "suspended", "2026-12-06", False, False),
        ("U-111", "marigold_joe", "2023-07-07", ["PO-2015"], "active", "", False, False),
        ("U-120", "deals4u_now", "2026-11-28", [], "active", "", False, False),
        ("U-121", "cheap_kicks_22", "2026-11-02", [], "active", "", False, False),
        ("U-122", "bargain_bill", "2024-06-10", [], "active", "", False, False),
        ("U-123", "promo_pia", "2026-11-20", [], "active", "", False, False),
        ("U-124", "flash_sale_fred", "2026-11-25", ["PO-2025"], "active", "", False, False),
        ("U-125", "leafy_lou", "2025-04-04", [], "active", "", False, False),
        ("U-130", "kuchen_klaus", "2023-10-10", [], "active", "", False, False),
        ("U-131", "herbst_hanna", "2024-09-09", [], "active", "", False, False),
        ("U-133", "gol_gabi", "2022-12-12", [], "active", "", False, False),
        ("U-134", "brot_bernd", "2025-01-01", ["PO-2034"], "active", "", False, False),
        ("U-140", "grimwald", "2021-06-06", ["PO-1940", "PO-1941", "PO-1942", "PO-1943"], "active", "", False, False),
        ("U-141", "grimwaldo", "2024-02-02", [], "active", "", False, False),
        ("U-150", "nina_reads", "2020-01-01", [], "active", "", False, False),
        ("U-151", "owl_at_noon", "2021-01-01", [], "active", "", False, False),
        ("U-152", "quiet_quill", "2021-03-01", [], "active", "", False, False),
        ("U-153", "ravi_r", "2022-01-01", [], "active", "", False, False),
        ("U-154", "sol_y_luna", "2022-05-01", [], "active", "", False, False),
        ("U-155", "tomte42", "2023-01-01", [], "active", "", False, False),
        ("U-156", "urban_fox", "2023-02-01", [], "active", "", False, False),
        ("U-157", "vela_v", "2023-03-01", [], "active", "", False, False),
        ("U-158", "wren_w", "2023-04-01", [], "active", "", False, False),
        ("U-159", "xan_x", "2023-05-01", [], "active", "", False, False),
        ("U-161", "late_bloomer", "2022-02-02", [], "active", "", False, False),
        ("U-162", "cartridge_carl", "2021-11-11", [], "active", "", False, False),
        ("U-163", "compost_kim", "2023-03-03", [], "active", "", False, False),
        ("U-164", "news_ned", "2020-10-10", [], "active", "", False, False),
        ("U-165", "tired_parent", "2024-04-04", [], "active", "", False, False),
        ("U-166", "metro_mo", "2022-06-06", [], "active", "", False, False),
        ("U-167", "rage_quit_ric", "2025-05-05", [], "active", "", False, False),
        ("U-170", "tidy_hunter", "2019-09-09", [], "active", "", False, True),
        ("U-180", "ultra_tomas", "2023-08-08", ["PO-1980"], "active", "", False, False),
        ("U-181", "barra_brava_9", "2024-07-07", ["PO-1981", "PO-1982"], "active", "", False, False),
        ("U-182", "la_pulga_fan", "2022-09-09", [], "active", "", False, False),
        ("U-183", "tiki_taka_tere", "2021-10-10", [], "active", "", False, False),
        ("U-184", "hincha_hugo", "2025-02-02", [], "active", "", False, False),
        ("U-185", "golazo_gus", "2023-11-11", [], "active", "", False, False),
        ("U-190", "misc_mia", "2022-01-15", [], "active", "", False, False),
    ]:
        users[uid] = {"handle": handle, "joined": joined, "strike_posts": strikes, "status": status,
                      "suspended_until": until, "verified": verified, "trusted_flagger": trusted}
    posts = {}
    for pid, author, comm, created, status, title in [
        ("PO-1901", "U-101", "c/retro-games", "09-10", "removed", "you all suck at speedruns"),
        ("PO-1902", "U-101", "c/retro-games", "10-05", "removed", "reply to a speedrun thread"),
        ("PO-2001", "U-101", "c/retro-games", "11-29", "visible", "comment on a high-score post"),
        ("PO-2002", "U-102", "c/retro-games", "11-30", "visible", "review of a homebrew cartridge"),
        ("PO-2003", "U-103", "c/retro-games", "11-28", "visible", "argument about emulators"),
        ("PO-2005", "U-104", "c/retro-games", "11-19", "visible", "pixel art thread"),
        ("PO-2006", "U-105", "c/local-news", "11-30", "visible", "council meeting recap"),
        ("PO-2010", "U-110", "c/gardening", "11-17", "removed", "seed swap offer"),
        ("PO-2011", "U-110", "c/gardening", "11-20", "removed", "link to a seed shop"),
        ("PO-2012", "U-110", "c/parenting", "11-24", "removed", "toy sale link"),
        ("PO-2015", "U-111", "c/gardening", "11-10", "removed", "compost bin advert"),
        ("PO-2020", "U-120", "c/local-news", "11-30", "visible", "cheap watches here"),
        ("PO-2021", "U-120", "c/local-news", "12-01", "visible", "more cheap watches"),
        ("PO-2022", "U-121", "c/local-news", "12-01", "visible", "sneaker discount"),
        ("PO-2023", "U-122", "c/retro-games", "11-30", "visible", "selling my console collection"),
        ("PO-2024", "U-123", "c/parenting", "12-01", "visible", "promo code for nappies"),
        ("PO-2025", "U-124", "c/local-news", "11-29", "removed", "flash sale"),
        ("PO-2027", "U-125", "c/gardening", "12-01", "visible", "reply in a pruning thread"),
        ("PO-2030", "U-130", "c/kochen", "11-27", "visible", "Rezept mit Link"),
        ("PO-2031", "U-131", "c/kochen", "11-26", "visible", "late-night post"),
        ("PO-2032", "U-102", "c/local-news", "11-28", "visible", "claim about the bypass"),
        ("PO-2033", "U-133", "c/futbol", "11-29", "visible", "entradas baratas"),
        ("PO-2034", "U-134", "c/kochen", "11-15", "removed", "Beleidigung"),
        ("PO-1940", "U-140", "c/local-news", "06-01", "removed", "slur"), ("PO-1941", "U-140", "c/local-news", "07-01", "removed", "slur"),
        ("PO-1942", "U-140", "c/local-news", "08-01", "removed", "slur"), ("PO-1943", "U-140", "c/local-news", "09-01", "removed", "slur"),
        ("PO-2040", "U-140", "c/local-news", "11-30", "visible", "rant about newcomers"),
        ("PO-2041", "U-140", "c/local-news", "11-29", "visible", "link to his blog"),
        ("PO-2045", "U-141", "c/local-news", "11-28", "visible", "bus timetable question"),
        ("PO-2050", "U-161", "c/local-news", "10-14", "visible", "old classified ad"),
        ("PO-2051", "U-162", "c/retro-games", "10-27", "visible", "rumour about a remake"),
        ("PO-2052", "U-163", "c/gardening", "10-31", "visible", "heated thread about slugs"),
        ("PO-2053", "U-164", "c/local-news", "11-01", "visible", "event listing"),
        ("PO-2054", "U-165", "c/parenting", "10-19", "visible", "I can't cope any more"),
        ("PO-2055", "U-166", "c/local-news", "10-04", "visible", "parking advert"),
        ("PO-2056", "U-167", "c/retro-games", "10-09", "visible", "angry post about a referee bot"),
        ("PO-2060", "U-190", "c/gardening", "11-24", "visible", "miracle fertiliser"),
        ("PO-2061", "U-163", "c/gardening", "11-19", "visible", "coffee grounds cure blight"),
        ("PO-2062", "U-125", "c/gardening", "11-28", "visible", "plant sale"),
        ("PO-2064", "U-165", "c/parenting", "11-23", "visible", "baby sleep course"),
        ("PO-2065", "U-161", "c/gardening", "11-14", "visible", "greenhouse for sale"),
        ("PO-1980", "U-180", "c/futbol", "09-20", "removed", "insulto"),
        ("PO-1981", "U-181", "c/futbol", "08-08", "removed", "insulto"), ("PO-1982", "U-181", "c/futbol", "10-10", "removed", "amenaza"),
        ("PO-2080", "U-182", "c/futbol", "11-19", "hidden", "opinion sobre el arbitro"),
        ("PO-2081", "U-180", "c/futbol", "11-21", "hidden", "ataque a otro usuario"),
        ("PO-2082", "U-183", "c/futbol", "11-21", "hidden", "comentario sobre la aficion rival"),
        ("PO-2083", "U-181", "c/futbol", "11-23", "hidden", "mensaje a un jugador"),
        ("PO-2085", "U-185", "c/futbol", "11-19", "visible", "venta de camisetas"),
        ("PO-2086", "U-130", "c/kochen", "11-20", "hidden", "Streit im Rezept-Thread"),
        ("PO-2087", "U-184", "c/futbol", "11-25", "hidden", "nos vemos en el estadio"),
        ("PO-2090", "U-190", "c/local-news", "11-27", "visible", "free sofa"),
        ("PO-2091", "U-162", "c/retro-games", "11-28", "visible", "fake leak"),
        ("PO-2092", "U-165", "c/parenting", "11-29", "visible", "school gate argument"),
        ("PO-2093", "U-164", "c/local-news", "11-26", "visible", "angry letter"),
        ("PO-2094", "U-131", "c/kochen", "11-27", "visible", "Kommentar"),
        ("PO-2095", "U-130", "c/kochen", "11-28", "visible", "Werbung"),
        ("PO-2096", "U-185", "c/futbol", "11-29", "visible", "oferta"),
        ("PO-2097", "U-164", "c/local-news", "11-29", "visible", "flood rumour"),
        ("PO-2098", "U-182", "c/futbol", "11-30", "visible", "cantico"),
    ]:
        posts[pid] = {"author": author, "community": comm, "created": "2026-" + created, "status": status, "title": title}
    reports = {}
    for rid, post, reporter, reason, created, status, assignee in [
        ("RP-3001", "PO-2001", "U-150", "harassment", "11-30", "open", ""),
        ("RP-3002", "PO-2001", "U-151", "harassment", "12-01", "open", ""),
        ("RP-3003", "PO-2002", "U-152", "harassment", "12-01", "open", ""),
        ("RP-3004", "PO-2003", "U-153", "harassment", "11-29", "open", ""),
        ("RP-3005", "PO-2003", "U-153", "harassment", "12-01", "open", ""),
        ("RP-3006", "PO-2006", "U-150", "harassment", "11-30", "open", ""),
        ("RP-3007", "PO-2006", "U-151", "harassment", "12-01", "open", ""),
        ("RP-3008", "PO-2005", "U-154", "spam", "11-25", "open", ""),
        ("RP-3009", "PO-2005", "U-155", "harassment", "11-20", "open", "M-01"),
        ("RP-3010", "PO-2010", "U-156", "spam", "11-18", "open", ""),
        ("RP-3011", "PO-2010", "U-157", "spam", "11-19", "open", "M-02"),
        ("RP-3012", "PO-2010", "U-158", "spam", "11-16", "actioned", "M-02"),
        ("RP-3013", "PO-2011", "U-156", "spam", "11-25", "open", ""),
        ("RP-3020", "PO-2020", "U-170", "spam", "12-01", "open", ""),
        ("RP-3021", "PO-2021", "U-170", "spam", "12-01", "open", ""),
        ("RP-3022", "PO-2022", "U-170", "spam", "12-02", "open", ""),
        ("RP-3023", "PO-2023", "U-170", "spam", "12-02", "open", ""),
        ("RP-3024", "PO-2024", "U-170", "spam", "12-02", "open", ""),
        ("RP-3025", "PO-2025", "U-170", "spam", "11-29", "actioned", "M-01"),
        ("RP-3027", "PO-2027", "U-170", "harassment", "12-02", "open", ""),
        ("RP-3030", "PO-2030", "U-159", "spam", "11-27", "open", "M-04"),
        ("RP-3031", "PO-2031", "U-152", "self_harm", "11-26", "escalated", "M-04"),
        ("RP-3032", "PO-2032", "U-153", "misinfo", "11-28", "open", "M-04"),
        ("RP-3033", "PO-2033", "U-154", "spam", "11-29", "open", "M-04"),
        ("RP-3034", "PO-2034", "U-155", "hate", "11-15", "actioned", "M-04"),
        ("RP-3040", "PO-2050", "U-156", "spam", "10-15", "open", ""),
        ("RP-3041", "PO-2051", "U-157", "misinfo", "10-28", "open", ""),
        ("RP-3042", "PO-2052", "U-158", "harassment", "11-01", "open", ""),
        ("RP-3043", "PO-2053", "U-159", "spam", "11-02", "open", ""),
        ("RP-3044", "PO-2054", "U-150", "self_harm", "10-20", "open", ""),
        ("RP-3045", "PO-2055", "U-151", "spam", "10-05", "open", "M-02"),
        ("RP-3046", "PO-2056", "U-152", "threat", "10-10", "open", ""),
        ("RP-3050", "PO-2040", "U-156", "hate", "12-01", "open", ""),
        ("RP-3051", "PO-2040", "U-157", "hate", "12-01", "open", ""),
        ("RP-3052", "PO-2041", "U-158", "spam", "11-30", "open", ""),
        ("RP-3060", "PO-2060", "U-150", "spam", "11-25", "open", ""),
        ("RP-3061", "PO-2060", "U-150", "spam", "11-27", "open", ""),
        ("RP-3062", "PO-2060", "U-151", "harassment", "11-28", "open", ""),
        ("RP-3063", "PO-2061", "U-152", "misinfo", "11-20", "open", "M-03"),
        ("RP-3064", "PO-2061", "U-153", "misinfo", "11-26", "open", ""),
        ("RP-3065", "PO-2061", "U-154", "misinfo", "11-30", "open", ""),
        ("RP-3066", "PO-2062", "U-155", "spam", "11-29", "open", ""),
        ("RP-3067", "PO-2062", "U-156", "spam", "11-30", "open", "M-02"),
        ("RP-3068", "PO-2064", "U-157", "spam", "11-24", "open", ""),
        ("RP-3069", "PO-2064", "U-158", "spam", "11-26", "open", ""),
        ("RP-3070", "PO-2065", "U-159", "spam", "11-15", "dismissed", ""),
        ("RP-3071", "PO-2065", "U-150", "spam", "11-28", "open", ""),
        ("RP-3080", "PO-2080", "U-151", "hate", "11-20", "dismissed", "M-08"),
        ("RP-3081", "PO-2080", "U-152", "hate", "11-21", "dismissed", "M-08"),
        ("RP-3082", "PO-2081", "U-153", "harassment", "11-22", "actioned", "M-02"),
        ("RP-3083", "PO-2081", "U-154", "harassment", "11-23", "dismissed", "M-02"),
        ("RP-3084", "PO-2082", "U-155", "hate", "11-22", "actioned", "M-09"),
        ("RP-3085", "PO-2082", "U-156", "hate", "11-25", "open", ""),
        ("RP-3086", "PO-2083", "U-157", "harassment", "11-24", "actioned", "M-09"),
        ("RP-3087", "PO-2087", "U-158", "threat", "11-26", "escalated", "M-09"),
        ("RP-3088", "PO-2085", "U-159", "spam", "11-20", "actioned", "M-02"),
        ("RP-3089", "PO-2086", "U-150", "hate", "11-21", "actioned", "M-07"),
        ("RP-3090", "PO-2090", "U-151", "spam", "11-28", "open", "M-02"),
        ("RP-3091", "PO-2091", "U-152", "misinfo", "11-29", "open", "M-03"),
        ("RP-3092", "PO-2092", "U-153", "harassment", "11-30", "open", "M-03"),
        ("RP-3093", "PO-2093", "U-154", "threat", "11-27", "escalated", "M-05"),
        ("RP-3094", "PO-2094", "U-155", "hate", "11-28", "open", "M-05"),
        ("RP-3095", "PO-2095", "U-156", "spam", "11-29", "open", "M-07"),
        ("RP-3096", "PO-2096", "U-157", "spam", "11-30", "open", "M-09"),
        ("RP-3097", "PO-2097", "U-158", "misinfo", "11-30", "open", "M-09"),
        ("RP-3098", "PO-2098", "U-159", "hate", "12-01", "escalated", "M-09"),
    ]:
        reports[rid] = {"post": post, "reporter": reporter, "reason": reason, "created": "2026-" + created,
                        "status": status, "assignee": assignee}
    mods = {}
    for mid, name, tier, langs, active, cap in [
        ("M-01", "Ada", 1, ["en"], True, 6), ("M-02", "Bruno", 1, ["en", "es"], True, 5),
        ("M-03", "Chen", 2, ["en"], True, 4), ("M-04", "Dagmar", 2, ["de", "en", "es"], True, 4),
        ("M-05", "Elif", 3, ["en", "de"], True, 3), ("M-06", "Femi", 1, ["en"], False, 5),
        ("M-07", "Greta", 1, ["de"], True, 4), ("M-08", "Hector", 2, ["es", "en"], True, 4),
        ("M-09", "Ingrid", 3, ["en", "es"], True, 5),
    ]:
        mods[mid] = {"name": name, "tier": tier, "languages": langs, "active": active, "max_open": cap}
    st = {"today": "2026-12-02",
          "policy": {"suspend_at": 3, "ban_at": 5, "suspend_days": 7, "new_account_days": 30},
          "communities": {"c/retro-games": "en", "c/local-news": "en", "c/gardening": "en", "c/parenting": "en",
                          "c/kochen": "de", "c/futbol": "es"},
          "users": users, "posts": posts, "reports": reports, "moderators": mods}
    env = ModerationEnv(st)
    for k, m in mods.items():
        assert env._load(k) <= m["max_open"], k
    for k, r in reports.items():
        if r["assignee"] and r["status"] in ("open", "escalated"):
            m = mods[r["assignee"]]
            assert env._lang(k) in m["languages"] and (m["tier"] >= 2 or not env._needs_high(k)), k
    for k, u in users.items():
        for p in u["strike_posts"]:
            assert posts[p]["author"] == k and posts[p]["status"] == "removed", (k, p)
    return st


_add("T25", "moderation", _mod(),
     "I'm Chen (M-03). Please triage the open, unassigned harassment reports in c/retro-games. Where a post has been "
     "reported for harassment by two or more different people, remove the post, give its author a strike for it and "
     "resolve those reports as actioned; if that strike brings the author to the suspension threshold, suspend them "
     "with me signing off. The harassment reports on any other post go to review: all reports on the same post to "
     "the same moderator, normally the tier 1 moderator with the fewest open reports, unless policy requires a "
     "higher tier, in which case the eligible moderator with the fewest open reports (ties: lower id). Other "
     "communities and other reasons are not part of this.",
     [["get_mod_policy", {}],
      ["list_reports", {"status": "open", "reason": "harassment", "community": "c/retro-games"}],
      ["list_moderators", {}], ["get_user", {"user_id": "U-101"}], ["get_user", {"user_id": "U-102"}],
      ["get_user", {"user_id": "U-103"}],
      ["set_post_status", {"post_id": "PO-2001", "status": "removed"}],
      ["add_strike", {"user_id": "U-101", "post_id": "PO-2001"}],
      ["resolve_report", {"report_id": "RP-3001", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3002", "outcome": "actioned"}],
      ["set_user_status", {"user_id": "U-101", "status": "suspended", "approved_by": "M-03"}],
      ["assign_report", {"report_id": "RP-3003", "moderator_id": "M-08"}],
      ["assign_report", {"report_id": "RP-3004", "moderator_id": "M-01"}],
      ["assign_report", {"report_id": "RP-3005", "moderator_id": "M-01"}]],
     [["set_post_status", {"post_id": "PO-2001", "status": "removed"}],
      ["add_strike", {"user_id": "U-101", "post_id": "PO-2001"}],
      ["resolve_report", {"report_id": "RP-3001", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3002", "outcome": "actioned"}],
      ["set_user_status", {"user_id": "U-101", "status": "suspended", "approved_by": "M-03"}],
      ["set_post_status", {"post_id": "PO-2003", "status": "removed"}],
      ["add_strike", {"user_id": "U-103", "post_id": "PO-2003"}],
      ["resolve_report", {"report_id": "RP-3004", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3005", "outcome": "actioned"}],
      ["assign_report", {"report_id": "RP-3003", "moderator_id": "M-08"}]],
     lambda s, i: _final(s, i, changes={
         "posts": {"PO-2001": {"status": "removed"}},
         "users": {"U-101": {"strike_posts": ["PO-1901", "PO-1902", "PO-2001"], "status": "suspended",
                             "suspended_until": "2026-12-09"}},
         "reports": {"RP-3001": {"status": "actioned"}, "RP-3002": {"status": "actioned"},
                     "RP-3003": {"assignee": "M-08"}, "RP-3004": {"assignee": "M-01"},
                     "RP-3005": {"assignee": "M-01"}}}),
     mentions=["c/retro-games"], seq=[[6, 7], [7, 10]])

_add("T26", "moderation", _mod(),
     "@marigold_jo appealed the removal of her post PO-2010 and the appeal has been upheld. Put the post back up, "
     "take away the strike she was given for it, and dismiss every report on it that is still open. If losing that "
     "strike takes her below the suspension threshold, lift her suspension, signed off by Dagmar (M-04); if not, "
     "the suspension stays. Her other removed posts, her other strikes and the reports on them stand as they are - "
     "and please don't confuse her with @marigold_joe.",
     [["get_mod_policy", {}], ["find_users", {"query": "marigold_jo"}], ["get_user", {"user_id": "U-110"}],
      ["get_post", {"post_id": "PO-2010"}],
      ["set_post_status", {"post_id": "PO-2010", "status": "visible"}],
      ["remove_strike", {"user_id": "U-110", "post_id": "PO-2010"}],
      ["resolve_report", {"report_id": "RP-3010", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3011", "outcome": "dismissed"}],
      ["set_user_status", {"user_id": "U-110", "status": "active", "approved_by": "M-04"}]],
     [["set_post_status", {"post_id": "PO-2010", "status": "visible"}],
      ["remove_strike", {"user_id": "U-110", "post_id": "PO-2010"}],
      ["resolve_report", {"report_id": "RP-3010", "outcome": "dismissed"}]],
     lambda s, i: _final(s, i, changes={
         "posts": {"PO-2010": {"status": "visible"}},
         "users": {"U-110": {"strike_posts": ["PO-2011", "PO-2012"], "status": "active", "suspended_until": ""}},
         "reports": {"RP-3010": {"status": "dismissed"}, "RP-3011": {"status": "dismissed"}}}),
     mentions=["marigold_jo", "marigold_joe", "Dagmar"], seq=[[5, 8]])


def _c27(s, i):
    opts = []
    for first in ("PO-2020", "PO-2021"):
        for a, b in (("M-02", "M-01"), ("M-01", "M-02")):
            opts.append(({"posts": {"PO-2020": {"status": "removed"}, "PO-2021": {"status": "removed"},
                                    "PO-2024": {"status": "removed"}},
                          "users": {"U-120": {"strike_posts": [first]}, "U-123": {"strike_posts": ["PO-2024"]}},
                          "reports": {"RP-3020": {"status": "actioned"}, "RP-3021": {"status": "actioned"},
                                      "RP-3024": {"status": "actioned"}, "RP-3022": {"assignee": a},
                                      "RP-3023": {"assignee": b}}}, None))
    return _any(*opts)(s, i)


_add("T27", "moderation", _mod(),
     "Our trusted flagger @tidy_hunter has reported a batch of posts as spam over the last two days. Go through "
     "those spam reports of theirs that are still open. Where policy lets us act on a trusted flagger's report "
     "without review, remove the post, resolve the report as actioned and give the author a strike - but only one "
     "strike per author for this whole batch, however many of their posts come down. Every other one of those "
     "reports goes to review: assign it to Bruno (M-02), or to Ada (M-01) once Bruno has no capacity left. Reports "
     "@tidy_hunter filed for other reasons are not part of this, and nothing else should change.",
     [["get_mod_policy", {}], ["find_users", {"query": "tidy_hunter"}],
      ["list_reports", {"reporter_id": "U-170", "status": "open", "reason": "spam"}],
      ["set_post_status", {"post_id": "PO-2020", "status": "removed"}],
      ["set_post_status", {"post_id": "PO-2021", "status": "removed"}],
      ["set_post_status", {"post_id": "PO-2024", "status": "removed"}],
      ["resolve_report", {"report_id": "RP-3020", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3021", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3024", "outcome": "actioned"}],
      ["add_strike", {"user_id": "U-120", "post_id": "PO-2020"}],
      ["add_strike", {"user_id": "U-123", "post_id": "PO-2024"}],
      ["assign_report", {"report_id": "RP-3022", "moderator_id": "M-02"}],
      ["assign_report", {"report_id": "RP-3023", "moderator_id": "M-01"}]],
     [["set_post_status", {"post_id": "PO-2020", "status": "removed"}],
      ["set_post_status", {"post_id": "PO-2021", "status": "removed"}],
      ["set_post_status", {"post_id": "PO-2022", "status": "removed"}],
      ["set_post_status", {"post_id": "PO-2024", "status": "removed"}],
      ["resolve_report", {"report_id": "RP-3020", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3021", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3022", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3024", "outcome": "actioned"}],
      ["add_strike", {"user_id": "U-120", "post_id": "PO-2020"}],
      ["add_strike", {"user_id": "U-121", "post_id": "PO-2022"}],
      ["add_strike", {"user_id": "U-123", "post_id": "PO-2024"}],
      ["assign_report", {"report_id": "RP-3023", "moderator_id": "M-02"}]],
     _c27, mentions=["tidy_hunter", "Bruno", "Ada"])

_add("T28", "moderation", _mod(),
     "Dagmar (M-04) is on leave from today. Hand every report she holds that is still open or escalated over to "
     "another moderator who is allowed to handle it - active, right language, right tier, with spare capacity - "
     "picking the one with the fewest open reports at that moment (ties go to the lower id). Escalated reports stay "
     "escalated. Reports she has already finished stay as they are, and no other assignment should change.",
     [["get_mod_policy", {}], ["list_reports", {"assignee": "M-04"}], ["list_moderators", {}],
      ["get_post", {"post_id": "PO-2032"}], ["get_user", {"user_id": "U-102"}],
      ["assign_report", {"report_id": "RP-3030", "moderator_id": "M-07"}],
      ["assign_report", {"report_id": "RP-3031", "moderator_id": "M-05"}],
      ["assign_report", {"report_id": "RP-3032", "moderator_id": "M-08"}],
      ["assign_report", {"report_id": "RP-3033", "moderator_id": "M-08"}]],
     [["assign_report", {"report_id": "RP-3030", "moderator_id": "M-07"}],
      ["assign_report", {"report_id": "RP-3031", "moderator_id": "M-05"}],
      ["assign_report", {"report_id": "RP-3032", "moderator_id": "M-01"}],
      ["assign_report", {"report_id": "RP-3033", "moderator_id": "M-08"}]],
     lambda s, i: _final(s, i, changes={"reports": {
         "RP-3030": {"assignee": "M-07"}, "RP-3031": {"assignee": "M-05"}, "RP-3032": {"assignee": "M-08"},
         "RP-3033": {"assignee": "M-08"}}}),
     mentions=["Dagmar"])

_add("T29", "moderation", _mod(),
     "Let's clear the backlog. Every open report that nobody has been assigned and that was filed more than 30 days "
     "ago should be dismissed, whatever its reason - threat reports included, since the senior-tier rule only "
     "matters while someone is handling a report. The only exception: if policy says a report like that may never "
     "be dismissed, escalate it instead, to "
     "the tier 2 or tier 3 moderator who works in the community's language and has the fewest open reports (ties: "
     "lower id). Reports filed 30 days ago or more recently, and reports someone is already working on, stay as "
     "they are, and so does everything else.",
     [["get_mod_policy", {}], ["list_reports", {"status": "open"}], ["list_moderators", {}],
      ["resolve_report", {"report_id": "RP-3046", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3040", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3044", "outcome": "dismissed"}],
      ["escalate_report", {"report_id": "RP-3044", "moderator_id": "M-08"}],
      ["resolve_report", {"report_id": "RP-3041", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3042", "outcome": "dismissed"}]],
     [["resolve_report", {"report_id": "RP-3046", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3040", "outcome": "dismissed"}],
      ["escalate_report", {"report_id": "RP-3044", "moderator_id": "M-08"}],
      ["resolve_report", {"report_id": "RP-3041", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3042", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3043", "outcome": "dismissed"}]],
     lambda s, i: _final(s, i, changes={"reports": {
         "RP-3046": {"status": "dismissed"}, "RP-3040": {"status": "dismissed"}, "RP-3041": {"status": "dismissed"},
         "RP-3042": {"status": "dismissed"}, "RP-3044": {"status": "escalated", "assignee": "M-08"}}}),
     ref_errors=[5])

_add("T30", "moderation", _mod(),
     "The hate-speech reports on @grimwald's post PO-2040 have been confirmed. Remove the post, resolve its open "
     "reports as actioned, and give him a strike for it. Then apply the strike thresholds in our policy to his "
     "account, with sign-off from a moderator who is allowed to give it - if more than one qualifies, the one with "
     "the fewest open reports. His other posts and the reports on them are not part of this, and @grimwaldo is a "
     "different person.",
     [["get_mod_policy", {}], ["find_users", {"query": "grimwald"}], ["get_user", {"user_id": "U-140"}],
      ["get_post", {"post_id": "PO-2040"}], ["list_moderators", {}],
      ["set_post_status", {"post_id": "PO-2040", "status": "removed"}],
      ["resolve_report", {"report_id": "RP-3050", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3051", "outcome": "actioned"}],
      ["add_strike", {"user_id": "U-140", "post_id": "PO-2040"}],
      ["set_user_status", {"user_id": "U-140", "status": "banned", "approved_by": "M-05"}]],
     [["set_post_status", {"post_id": "PO-2040", "status": "removed"}],
      ["resolve_report", {"report_id": "RP-3050", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3051", "outcome": "actioned"}],
      ["add_strike", {"user_id": "U-140", "post_id": "PO-2040"}],
      ["set_user_status", {"user_id": "U-140", "status": "suspended", "approved_by": "M-03"}]],
     lambda s, i: _final(s, i, changes={
         "posts": {"PO-2040": {"status": "removed"}},
         "users": {"U-140": {"strike_posts": ["PO-1940", "PO-1941", "PO-1942", "PO-1943", "PO-2040"],
                             "status": "banned"}},
         "reports": {"RP-3050": {"status": "actioned"}, "RP-3051": {"status": "actioned"}}}),
     mentions=["grimwald", "grimwaldo"], seq=[[5, 8], [8, 9]])

_add("T31", "moderation", _mod(),
     "Tidy up the open reports in c/gardening. Where one post has several open reports for the same reason, keep "
     "the oldest of them and dismiss the newer ones as duplicates - except that a report already assigned to a "
     "moderator is being worked on and stays open (it still counts when deciding which report is the oldest). "
     "Reports on the same post for a different reason are not duplicates. Separately, hide every c/gardening post "
     "that had open reports from three or more different people before you started. Nothing else should change.",
     [["get_mod_policy", {}], ["list_reports", {"community": "c/gardening", "status": "open"}],
      ["list_reports", {"community": "c/gardening", "status": "open", "page": 2}],
      ["resolve_report", {"report_id": "RP-3061", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3064", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3065", "outcome": "dismissed"}],
      ["set_post_status", {"post_id": "PO-2061", "status": "hidden"}]],
     [["resolve_report", {"report_id": "RP-3061", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3064", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3065", "outcome": "dismissed"}],
      ["set_post_status", {"post_id": "PO-2060", "status": "hidden"}],
      ["set_post_status", {"post_id": "PO-2061", "status": "hidden"}]],
     lambda s, i: _final(s, i, changes={
         "reports": {"RP-3061": {"status": "dismissed"}, "RP-3064": {"status": "dismissed"},
                     "RP-3065": {"status": "dismissed"}},
         "posts": {"PO-2061": {"status": "hidden"}}}),
     mentions=["c/gardening"])

_add("T32", "moderation", _mod(),
     "Please review the hidden posts in c/futbol, one by one. First, if any report on the post is still open or "
     "escalated, leave the post hidden for now. Otherwise: if every report on it was dismissed, make it visible "
     "again; if at least one was actioned, remove the post and give its author a strike for it, and if that brings "
     "the author to the suspension threshold, suspend them with Hector (M-08) signing off. Visible posts, removed "
     "posts and other communities are not part of this, and the reports themselves stay as they are.",
     [["get_mod_policy", {}], ["list_posts", {"community": "c/futbol", "status": "hidden"}],
      ["get_post", {"post_id": "PO-2080"}], ["get_post", {"post_id": "PO-2081"}], ["get_post", {"post_id": "PO-2082"}],
      ["get_post", {"post_id": "PO-2083"}], ["get_post", {"post_id": "PO-2087"}],
      ["set_post_status", {"post_id": "PO-2080", "status": "visible"}],
      ["set_post_status", {"post_id": "PO-2081", "status": "removed"}],
      ["add_strike", {"user_id": "U-180", "post_id": "PO-2081"}],
      ["set_post_status", {"post_id": "PO-2083", "status": "removed"}],
      ["add_strike", {"user_id": "U-181", "post_id": "PO-2083"}],
      ["set_user_status", {"user_id": "U-181", "status": "suspended", "approved_by": "M-08"}]],
     [["set_post_status", {"post_id": "PO-2080", "status": "visible"}],
      ["set_post_status", {"post_id": "PO-2081", "status": "removed"}],
      ["add_strike", {"user_id": "U-180", "post_id": "PO-2081"}],
      ["set_post_status", {"post_id": "PO-2082", "status": "removed"}],
      ["add_strike", {"user_id": "U-183", "post_id": "PO-2082"}],
      ["set_post_status", {"post_id": "PO-2083", "status": "removed"}],
      ["add_strike", {"user_id": "U-181", "post_id": "PO-2083"}],
      ["set_user_status", {"user_id": "U-181", "status": "suspended", "approved_by": "M-08"}]],
     lambda s, i: _final(s, i, changes={
         "posts": {"PO-2080": {"status": "visible"}, "PO-2081": {"status": "removed"},
                   "PO-2083": {"status": "removed"}},
         "users": {"U-180": {"strike_posts": ["PO-1980", "PO-2081"]},
                   "U-181": {"strike_posts": ["PO-1981", "PO-1982", "PO-2083"], "status": "suspended",
                             "suspended_until": "2026-12-09"}}}),
     mentions=["c/futbol", "Hector"], seq=[[8, 9], [10, 11, 12]])


# --------------------------------------------------------------------------
# 5. exam invigilation: exams, rooms, staff, invigilator assignments
# --------------------------------------------------------------------------
class ExamsEnv(_Env):
    TOOLS = [
        _fn("get_exam_policy", "Today's date and the invigilation and room rules that the write tools enforce."),
        _fn("list_exams", "Exams, filtered, by date, start time and id; 8 per page. Each row lists the current "
            "invigilators and the number policy requires.",
            {"date": _s("Only this date, YYYY-MM-DD."), "subject": _s("Only this subject."),
             "room_id": _s("Only exams in this room."), "page": _PAGE}),
        _fn("get_exam", "One exam in detail, with its invigilators and assignment ids.",
            {"exam_id": _s("Exam id, e.g. EX-101.")}, ["exam_id"]),
        _fn("list_rooms", "All exam rooms with capacity and the dates they are closed."),
        _fn("find_staff", "Staff who can invigilate, filtered by name (case-insensitive substring), subject taught "
            "and/or lead qualification. With no filter, everyone.",
            {"query": _s("Text that must appear in the name."), "subject": _s("Only teachers of this subject."),
             "lead_qualified": _b("Only staff with (true) or without (false) the lead qualification.")}),
        _fn("list_staff_schedule", "For one date: every member of staff, whether they are available, and the exam "
            "sessions they invigilate that day.", {"date": _s("Date, YYYY-MM-DD.")}, ["date"]),
        _fn("assign_invigilator", "Add a member of staff to an exam as lead or support invigilator.",
            {"exam_id": _s("Exam id."), "staff_id": _s("Staff id, e.g. ST-01."),
             "role": _s("Role.", ["lead", "support"])}, ["exam_id", "staff_id", "role"]),
        _fn("remove_invigilator", "Take a member of staff off an exam.",
            {"exam_id": _s("Exam id."), "staff_id": _s("Staff id.")}, ["exam_id", "staff_id"]),
        _fn("set_invigilator_role", "Change the role of an invigilator already on an exam.",
            {"exam_id": _s("Exam id."), "staff_id": _s("Staff id."), "role": _s("New role.", ["lead", "support"])},
            ["exam_id", "staff_id", "role"]),
        _fn("set_exam_room", "Move an exam to another room (same date and time).",
            {"exam_id": _s("Exam id."), "room_id": _s("Room id, e.g. RM-01.")}, ["exam_id", "room_id"]),
        _fn("set_exam_candidates", "Change the number of candidates sitting an exam.",
            {"exam_id": _s("Exam id."), "candidates": _i("New number of candidates.", 1)}, ["exam_id", "candidates"]),
        _fn("reschedule_exam", "Move an exam to another date, start time and room.",
            {"exam_id": _s("Exam id."), "date": _s("New date, YYYY-MM-DD."), "start": _s("New start time, HH:MM."),
             "room_id": _s("Room id.")}, ["exam_id", "date", "start", "room_id"]),
        _fn("split_exam", "Move some of an exam's candidates into a second sitting of the same paper, at the same "
            "date and time, in another room. The new sitting starts with no invigilators.",
            {"exam_id": _s("Exam id."), "room_id": _s("Room for the new sitting."),
             "candidates": _i("Number of candidates moved to the new sitting.", 1)}, ["exam_id", "room_id", "candidates"]),
        _fn("set_staff_unavailable", "Record that a member of staff is unavailable on a date.",
            {"staff_id": _s("Staff id."), "date": _s("Date, YYYY-MM-DD.")}, ["staff_id", "date"]),
    ]

    def _required(self, ex):
        p = self.state["policy"]
        return max(p["min_invigilators"], -(-ex["candidates"] // p["candidates_per_invigilator"]))

    def _span(self, ex):
        s0 = _ptime(ex["start"])
        return s0, s0 + ex["duration"]

    def _active(self, exam_id=None, staff_id=None):
        return [k for k, x in sorted(self.state["assignments"].items()) if x["status"] == "active"
                and (exam_id is None or x["exam"] == exam_id) and (staff_id is None or x["staff"] == staff_id)]

    def _sessions(self, staff_id, date):
        return [self.state["assignments"][k]["exam"] for k in self._active(staff_id=staff_id)
                if self.state["exams"][self.state["assignments"][k]["exam"]]["date"] == date]

    def _overlap(self, a, b):
        if a["date"] != b["date"]:
            return False
        a0, a1 = self._span(a)
        b0, b1 = self._span(b)
        return a0 < b1 and b0 < a1

    def _exam_row(self, k):
        ex = self.state["exams"][k]
        s0, s1 = self._span(ex)
        inv = []
        for a in self._active(exam_id=k):
            x = self.state["assignments"][a]
            inv.append({"assignment_id": a, "staff_id": x["staff"], "name": self.state["staff"][x["staff"]]["name"],
                        "role": x["role"]})
        return {"exam_id": k, "paper": ex["paper"], "subject": ex["subject"], "date": ex["date"], "start": ex["start"],
                "end": _fmt_time(s1), "room_id": ex["room"], "candidates": ex["candidates"],
                "invigilators_required": self._required(ex), "invigilators": inv, "sitting_of": ex["sitting_of"]}

    def _exam(self, k):
        ex = self._get("exams", k, "exam")
        if ex["status"] != "scheduled":
            raise _Rule("exam %s is %s" % (k, ex["status"]))
        return ex

    def _room_ok(self, exam_id, room_id, date, start, duration, candidates):
        r = self._get("rooms", room_id, "room")
        if candidates > r["capacity"]:
            raise _Rule("%s (%s) seats %d; %d candidates do not fit" % (room_id, r["name"], r["capacity"], candidates))
        if date in r["closed_dates"]:
            raise _Rule("%s (%s) is closed on %s" % (room_id, r["name"], date))
        probe = {"date": date, "start": start, "duration": duration}
        for k, ex in sorted(self.state["exams"].items()):
            if k != exam_id and ex["status"] == "scheduled" and ex["room"] == room_id and self._overlap(ex, probe):
                raise _Rule("%s is in use by %s (%s %s) at that time" % (room_id, k, ex["paper"], ex["start"]))

    def _t_get_exam_policy(self, a):
        p = self.state["policy"]
        return {"today": self.state["today"], "policy": {
            "candidates_per_invigilator": p["candidates_per_invigilator"], "min_invigilators": p["min_invigilators"],
            "max_sessions_per_day": p["max_sessions_per_day"],
            "rules": [
                "Invigilators required per exam = candidates / candidates_per_invigilator, rounded up, but never fewer "
                "than min_invigilators. An exam may not be given more invigilators than it requires.",
                "Every exam has exactly one lead invigilator, who must hold the lead qualification; all others are "
                "support. While an exam has no lead, its last free place is reserved for a lead.",
                "A teacher may not invigilate an exam in the subject they teach.",
                "Staff cannot invigilate on a date they are unavailable, in two exams whose times overlap, or in more "
                "than max_sessions_per_day exams on one day.",
                "An exam's room must seat all its candidates, be open that day and not be used by another exam at an "
                "overlapping time.",
                "An exam can only be rescheduled once all its invigilators have been taken off it.",
                "A member of staff can only be recorded as unavailable on a date when they have no exams that day.",
            ]}}

    def _t_list_exams(self, a):
        if "date" in a:
            _pdate(a["date"])
        if "room_id" in a:
            self._get("rooms", a["room_id"], "room")
        rows = [k for _, _, k in sorted((ex["date"], ex["start"], k) for k, ex in self.state["exams"].items()
                                        if ex["status"] == "scheduled"
                                        and ("date" not in a or ex["date"] == a["date"])
                                        and ("subject" not in a or ex["subject"].lower() == a["subject"].strip().lower())
                                        and ("room_id" not in a or ex["room"] == a["room_id"]))]
        meta, rows = _paged(rows, a.get("page", 1), 8)
        meta["exams"] = [self._exam_row(k) for k in rows]
        return meta

    def _t_get_exam(self, a):
        self._get("exams", a["exam_id"], "exam")
        row = self._exam_row(a["exam_id"])
        row["duration_minutes"] = self.state["exams"][a["exam_id"]]["duration"]
        row["room_name"] = self.state["rooms"][row["room_id"]]["name"]
        return row

    def _t_list_rooms(self, a):
        return {"rooms": [dict(room_id=k, **r) for k, r in sorted(self.state["rooms"].items())]}

    def _t_find_staff(self, a):
        q = a.get("query", "").strip().lower()
        return {"staff": [{"staff_id": k, "name": x["name"], "teaches": x["subject"] or "(non-teaching)",
                           "lead_qualified": x["lead"], "unavailable": list(x["unavailable"])}
                          for k, x in sorted(self.state["staff"].items())
                          if q in x["name"].lower()
                          and ("subject" not in a or x["subject"].lower() == a["subject"].strip().lower())
                          and ("lead_qualified" not in a or x["lead"] == a["lead_qualified"])]}

    def _t_list_staff_schedule(self, a):
        _pdate(a["date"])
        out = []
        for k, x in sorted(self.state["staff"].items()):
            sess = []
            for ak in self._active(staff_id=k):
                asg = self.state["assignments"][ak]
                ex = self.state["exams"][asg["exam"]]
                if ex["date"] == a["date"]:
                    s0, s1 = self._span(ex)
                    sess.append({"exam_id": asg["exam"], "paper": ex["paper"], "start": ex["start"],
                                 "end": _fmt_time(s1), "role": asg["role"]})
            sess.sort(key=lambda r: r["start"])
            out.append({"staff_id": k, "name": x["name"], "teaches": x["subject"] or "(non-teaching)",
                        "lead_qualified": x["lead"], "available": a["date"] not in x["unavailable"],
                        "sessions": sess})
        return {"date": a["date"], "staff": out}

    def _t_assign_invigilator(self, a):
        ex = self._exam(a["exam_id"])
        st = self._get("staff", a["staff_id"], "staff member")
        if self._active(a["exam_id"], a["staff_id"]):
            raise _Rule("%s is already invigilating %s" % (a["staff_id"], a["exam_id"]))
        if ex["date"] in st["unavailable"]:
            raise _Rule("%s is unavailable on %s" % (a["staff_id"], ex["date"]))
        if st["subject"] and st["subject"].lower() == ex["subject"].lower():
            raise _Rule("%s teaches %s and may not invigilate %s" % (a["staff_id"], st["subject"], ex["paper"]))
        mine = self._sessions(a["staff_id"], ex["date"])
        for k in mine:
            if self._overlap(self.state["exams"][k], ex):
                raise _Rule("%s is already invigilating %s at an overlapping time" % (a["staff_id"], k))
        if len(mine) >= self.state["policy"]["max_sessions_per_day"]:
            raise _Rule("%s already has %d sessions on %s" % (a["staff_id"], len(mine), ex["date"]))
        act = self._active(exam_id=a["exam_id"])
        need = self._required(ex)
        if len(act) >= need:
            raise _Rule("%s already has %d invigilators; it requires %d" % (a["exam_id"], len(act), need))
        has_lead = any(self.state["assignments"][k]["role"] == "lead" for k in act)
        if a["role"] == "lead":
            if not st["lead"]:
                raise _Rule("%s does not hold the lead qualification" % a["staff_id"])
            if has_lead:
                raise _Rule("%s already has a lead invigilator" % a["exam_id"])
        elif not has_lead and len(act) == need - 1:
            raise _Rule("the last place on %s is reserved for a lead invigilator" % a["exam_id"])
        aid = self._new_id("next_assign", "AS")
        self.state["assignments"][aid] = {"exam": a["exam_id"], "staff": a["staff_id"], "role": a["role"],
                                          "status": "active"}
        return {"ok": True, "assignment_id": aid, "exam": self._exam_row(a["exam_id"])}

    def _t_remove_invigilator(self, a):
        self._exam(a["exam_id"])
        self._get("staff", a["staff_id"], "staff member")
        act = self._active(a["exam_id"], a["staff_id"])
        if not act:
            raise _Rule("%s is not invigilating %s" % (a["staff_id"], a["exam_id"]))
        self.state["assignments"][act[0]]["status"] = "removed"
        return {"ok": True, "exam": self._exam_row(a["exam_id"])}

    def _t_set_invigilator_role(self, a):
        self._exam(a["exam_id"])
        st = self._get("staff", a["staff_id"], "staff member")
        act = self._active(a["exam_id"], a["staff_id"])
        if not act:
            raise _Rule("%s is not invigilating %s" % (a["staff_id"], a["exam_id"]))
        x = self.state["assignments"][act[0]]
        if x["role"] == a["role"]:
            raise _Rule("%s is already %s on %s" % (a["staff_id"], a["role"], a["exam_id"]))
        if a["role"] == "lead":
            if not st["lead"]:
                raise _Rule("%s does not hold the lead qualification" % a["staff_id"])
            others = [k for k in self._active(exam_id=a["exam_id"]) if self.state["assignments"][k]["role"] == "lead"]
            if others:
                raise _Rule("%s already has a lead (%s); make them support first" % (
                    a["exam_id"], self.state["assignments"][others[0]]["staff"]))
        x["role"] = a["role"]
        return {"ok": True, "exam": self._exam_row(a["exam_id"])}

    def _t_set_exam_room(self, a):
        ex = self._exam(a["exam_id"])
        if ex["room"] == a["room_id"]:
            raise _Rule("%s is already in %s" % (a["exam_id"], a["room_id"]))
        self._room_ok(a["exam_id"], a["room_id"], ex["date"], ex["start"], ex["duration"], ex["candidates"])
        ex["room"] = a["room_id"]
        return {"ok": True, "exam": self._exam_row(a["exam_id"])}

    def _t_set_exam_candidates(self, a):
        ex = self._exam(a["exam_id"])
        if ex["candidates"] == a["candidates"]:
            raise _Rule("%s already has %d candidates" % (a["exam_id"], ex["candidates"]))
        cap = self.state["rooms"][ex["room"]]["capacity"]
        if a["candidates"] > cap:
            raise _Rule("%s seats only %d" % (ex["room"], cap))
        ex["candidates"] = a["candidates"]
        return {"ok": True, "exam": self._exam_row(a["exam_id"])}

    def _t_reschedule_exam(self, a):
        ex = self._exam(a["exam_id"])
        if self._active(exam_id=a["exam_id"]):
            raise _Rule("%s still has invigilators; take them off before rescheduling" % a["exam_id"])
        if _pdate(a["date"]) <= _pdate(self.state["today"]):
            raise _Rule("exams can only be moved to a date after today (%s)" % self.state["today"])
        _ptime(a["start"], "start")
        self._room_ok(a["exam_id"], a["room_id"], a["date"], a["start"], ex["duration"], ex["candidates"])
        ex["date"], ex["start"], ex["room"] = a["date"], a["start"], a["room_id"]
        return {"ok": True, "exam": self._exam_row(a["exam_id"])}

    def _t_split_exam(self, a):
        ex = self._exam(a["exam_id"])
        if a["candidates"] >= ex["candidates"]:
            raise _Rule("%s has %d candidates; a split must leave some behind" % (a["exam_id"], ex["candidates"]))
        if a["room_id"] == ex["room"]:
            raise _Rule("the new sitting needs a different room")
        self._room_ok(None, a["room_id"], ex["date"], ex["start"], ex["duration"], a["candidates"])
        nid = self._new_id("next_exam", "EX")
        self.state["exams"][nid] = {"paper": ex["paper"], "subject": ex["subject"], "date": ex["date"],
                                    "start": ex["start"], "duration": ex["duration"], "room": a["room_id"],
                                    "candidates": a["candidates"], "status": "scheduled", "sitting_of": a["exam_id"]}
        ex["candidates"] -= a["candidates"]
        return {"ok": True, "new_exam": self._exam_row(nid), "original": self._exam_row(a["exam_id"])}

    def _t_set_staff_unavailable(self, a):
        st = self._get("staff", a["staff_id"], "staff member")
        _pdate(a["date"])
        if a["date"] in st["unavailable"]:
            raise _Rule("%s is already unavailable on %s" % (a["staff_id"], a["date"]))
        busy = self._sessions(a["staff_id"], a["date"])
        if busy:
            raise _Rule("%s still invigilates %s on %s" % (a["staff_id"], ", ".join(busy), a["date"]))
        st["unavailable"].append(a["date"])
        st["unavailable"].sort()
        return {"ok": True, "staff_id": a["staff_id"], "unavailable": list(st["unavailable"])}


def _exams():
    rooms = {}
    for rid, name, cap, closed in [
        ("RM-01", "Main Hall", 120, []), ("RM-02", "Sports Hall", 100, []), ("RM-03", "Library", 40, []),
        ("RM-04", "Lecture Theatre", 60, []), ("RM-05", "Seminar Room A", 16, []), ("RM-06", "Seminar Room B", 24, []),
        ("RM-07", "Music Studio", 20, ["2027-01-12"]), ("RM-08", "Drama Studio", 35, []),
    ]:
        rooms[rid] = {"name": name, "capacity": cap, "closed_dates": closed}
    staff = {}
    for sid, name, subject, lead, unav in [
        ("ST-01", "Joan Pell", "", True, ["2027-01-15"]), ("ST-02", "Victor Lam", "", True, []),
        ("ST-03", "Sana Mirza", "", False, []), ("ST-04", "Bill Hart", "", False, []),
        ("ST-05", "Clara Diaz", "Physics", True, ["2027-01-13", "2027-01-15"]), ("ST-06", "Noel Grant", "Physics", False, []),
        ("ST-07", "Femi Okoro", "Chemistry", True, ["2027-01-15"]), ("ST-08", "Lotte Berg", "Chemistry", False, []),
        ("ST-09", "Paul Weir", "Biology", True, ["2027-01-15"]), ("ST-10", "Asha Rao", "Biology", False, []),
        ("ST-11", "Tom Quist", "Maths", True, ["2027-01-15"]), ("ST-12", "Eva Novak", "Maths", False, []),
        ("ST-13", "Ian Moss", "History", False, []), ("ST-14", "Gwen Tate", "English", True, []),
        ("ST-15", "Omar Saleh", "", False, []), ("ST-16", "Kit Byrne", "", True, []),
        ("ST-17", "Mina Cho", "Geography", False, []), ("ST-18", "Dara Flynn", "", False, ["2027-01-12"]),
        ("ST-19", "Zara Malik", "English", False, []), ("ST-20", "Hugo Varga", "", False, []),
        ("ST-21", "Rita Varga", "Physics", True, []), ("ST-22", "Leo Park", "", False, []),
    ]:
        staff[sid] = {"name": name, "subject": subject, "lead": lead, "unavailable": unav}
    exams = {}
    assignments = {}
    n = [500]
    for eid, paper, subject, date, start, dur, room, cands, inv in [
        ("EX-101", "Maths Paper 1", "Maths", "01-11", "09:00", 120, "RM-01", 110,
         [("ST-14", "lead"), ("ST-12", "support"), ("ST-16", "support"), ("ST-03", "support"), ("ST-19", "support")]),
        ("EX-102", "History Paper 1", "History", "01-11", "09:00", 90, "RM-04", 48, [("ST-02", "lead"), ("ST-01", "lead")]),
        ("EX-103", "English Literature Paper 1", "English", "01-11", "13:30", 120, "RM-01", 95,
         [("ST-11", "lead"), ("ST-15", "support"), ("ST-20", "support"), ("ST-04", "support"), ("ST-22", "support")]),
        ("EX-104", "Geography Paper 1", "Geography", "01-11", "13:30", 90, "RM-03", 38,
         [("ST-05", "lead"), ("ST-13", "support")]),
        ("EX-111", "Physics Paper 2", "Physics", "01-12", "09:00", 120, "RM-04", 85, []),
        ("EX-112", "Biology Paper 1", "Biology", "01-12", "09:00", 120, "RM-01", 140,
         [("ST-01", "lead"), ("ST-04", "support"), ("ST-16", "support"), ("ST-22", "support"), ("ST-12", "support"),
          ("ST-13", "support")]),
        ("EX-113", "Chemistry Paper 2", "Chemistry", "01-12", "13:30", 120, "RM-02", 70,
         [("ST-02", "lead"), ("ST-03", "support"), ("ST-15", "support")]),
        ("EX-114", "Computing Paper 1", "Computing", "01-12", "13:30", 90, "RM-03", 30,
         [("ST-14", "lead"), ("ST-20", "support")]),
        ("EX-121", "Chemistry Paper 1", "Chemistry", "01-13", "09:00", 120, "RM-01", 100,
         [("ST-16", "lead"), ("ST-03", "support"), ("ST-19", "support"), ("ST-20", "support")]),
        ("EX-122", "French Paper 1", "French", "01-13", "09:00", 90, "RM-04", 40, [("ST-07", "lead"), ("ST-15", "support")]),
        ("EX-123", "Maths Paper 2", "Maths", "01-13", "13:30", 120, "RM-01", 110,
         [("ST-09", "lead"), ("ST-07", "support"), ("ST-01", "support"), ("ST-04", "support"), ("ST-22", "support")]),
        ("EX-124", "Art History Paper 1", "Art", "01-13", "13:30", 90, "RM-08", 28, [("ST-02", "lead"), ("ST-17", "support")]),
        ("EX-131", "Physics Paper 1", "Physics", "01-14", "09:00", 120, "RM-02", 88,
         [("ST-14", "lead"), ("ST-03", "support"), ("ST-13", "support"), ("ST-20", "support")]),
        ("EX-132", "Economics Paper 1", "Economics", "01-14", "09:00", 90, "RM-03", 36,
         [("ST-02", "lead"), ("ST-07", "support")]),
        ("EX-133", "Biology Paper 2", "Biology", "01-14", "13:30", 120, "RM-02", 75,
         [("ST-16", "lead"), ("ST-04", "support"), ("ST-15", "support")]),
        ("EX-134", "English Language Paper 1", "English", "01-14", "13:30", 120, "RM-01", 115,
         [("ST-11", "lead"), ("ST-01", "support"), ("ST-22", "support"), ("ST-12", "support"), ("ST-17", "support")]),
        ("EX-140", "Music Paper 1", "Music", "01-15", "09:00", 90, "RM-01", 30, [("ST-14", "lead"), ("ST-19", "support")]),
        ("EX-141", "German Paper 1", "German", "01-15", "09:00", 90, "RM-04", 44, [("ST-02", "lead"), ("ST-21", "support")]),
        ("EX-142", "History Paper 2", "History", "01-15", "13:30", 120, "RM-01", 70,
         [("ST-21", "lead"), ("ST-08", "support"), ("ST-03", "support")]),
        ("EX-143", "Physics Paper 3", "Physics", "01-15", "13:30", 90, "RM-03", 34, [("ST-16", "lead"), ("ST-20", "support")]),
        ("EX-144", "Spanish Paper 1", "Spanish", "01-15", "13:30", 90, "RM-04", 40, [("ST-14", "lead"), ("ST-12", "support")]),
        ("EX-145", "Drama Paper 1", "Drama", "01-15", "13:30", 60, "RM-08", 18, [("ST-02", "lead"), ("ST-17", "support")]),
    ]:
        exams[eid] = {"paper": paper, "subject": subject, "date": "2027-" + date, "start": start, "duration": dur,
                      "room": room, "candidates": cands, "status": "scheduled", "sitting_of": ""}
        for sid, role in inv:
            n[0] += 1
            assignments["AS-%d" % n[0]] = {"exam": eid, "staff": sid, "role": role, "status": "active"}
    st = {"today": "2027-01-08",
          "policy": {"candidates_per_invigilator": 25, "min_invigilators": 2, "max_sessions_per_day": 2},
          "rooms": rooms, "staff": staff, "exams": exams, "assignments": assignments,
          "next_exam": 900, "next_assign": 600}
    # sanity: everything obeys the rules except the problems the tasks are about
    env = ExamsEnv(st)
    planted = {"EX-101", "EX-102", "EX-103", "EX-111", "EX-112"}
    for k, ex in exams.items():
        act = env._active(exam_id=k)
        leads = [x for x in act if assignments[x]["role"] == "lead"]
        if k not in planted:
            assert ex["candidates"] <= rooms[ex["room"]]["capacity"], k
            assert len(act) in (0, env._required(ex)), (k, len(act), env._required(ex))
            assert len(leads) == (1 if act else 0), k
            for x in act:
                assert staff[assignments[x]["staff"]]["subject"] != ex["subject"], (k, x)
        for x in act:
            assert ex["date"] not in staff[assignments[x]["staff"]]["unavailable"], (k, x)
            assert assignments[x]["role"] != "lead" or staff[assignments[x]["staff"]]["lead"], (k, x)
        for k2, ex2 in exams.items():
            assert k2 == k or ex2["room"] != ex["room"] or not env._overlap(ex, ex2), (k, k2)
    for sid in staff:
        for d in set(ex["date"] for ex in exams.values()):
            sess = env._sessions(sid, d)
            assert len(sess) <= 2, (sid, d)
            for x, y in itertools.combinations(sess, 2):
                assert not env._overlap(exams[x], exams[y]), (sid, x, y)
    return st


def _as_id(i, exam, staff_id):
    ks = [k for k, x in i["assignments"].items() if x["exam"] == exam and x["staff"] == staff_id and x["status"] == "active"]
    assert len(ks) == 1, (exam, staff_id)
    return ks[0]


def _gone(i, pairs):
    return dict((_as_id(i, e, s), {"status": "removed"}) for e, s in pairs)


def _asg(exam, staff_id, role):
    return {"exam": exam, "staff": staff_id, "role": role, "status": "active"}


_add("T33", "exams", _exams(),
     "Physics Paper 2 on Tuesday 12 January has no invigilators yet, and I've just noticed its room is too small for "
     "the number of candidates. Move it to the smallest room that can seat everyone and is free at that time, then "
     "give it exactly the number of invigilators our policy requires, with a qualified lead. Use non-teaching staff "
     "only, unless there aren't enough of them free, in which case top up with teachers. Don't change any other "
     "exam or anyone else's assignments.",
     [["get_exam_policy", {}], ["list_exams", {"date": "2027-01-12"}], ["list_rooms", {}],
      ["set_exam_room", {"exam_id": "EX-111", "room_id": "RM-02"}],
      ["list_staff_schedule", {"date": "2027-01-12"}],
      ["assign_invigilator", {"exam_id": "EX-111", "staff_id": "ST-02", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-111", "staff_id": "ST-03", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-111", "staff_id": "ST-15", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-111", "staff_id": "ST-20", "role": "support"}]],
     [["set_exam_room", {"exam_id": "EX-111", "room_id": "RM-02"}],
      ["assign_invigilator", {"exam_id": "EX-111", "staff_id": "ST-02", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-111", "staff_id": "ST-03", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-111", "staff_id": "ST-15", "role": "support"}]],
     lambda s, i: _final(s, i, changes={"exams": {"EX-111": {"room": "RM-02"}}}, new={"assignments": [
         _asg("EX-111", "ST-02", "lead"), _asg("EX-111", "ST-03", "support"), _asg("EX-111", "ST-15", "support"),
         _asg("EX-111", "ST-20", "support")]}),
     mentions=["Physics Paper 2"])

_add("T34", "exams", _exams(),
     "Femi Okoro has called in sick for Wednesday 13 and Thursday 14 January. Take him off every exam he was "
     "invigilating on those two days and record him as unavailable on both dates. Replace him in each of those "
     "exams with an eligible member of staff in the same role - if he was the lead, the replacement must be "
     "qualified to lead - choosing whoever has the fewest exam sessions that day (ties: lower staff id). His "
     "assignments on other days stay, and nothing else should change.",
     [["get_exam_policy", {}], ["find_staff", {"query": "Okoro"}],
      ["list_staff_schedule", {"date": "2027-01-13"}], ["list_staff_schedule", {"date": "2027-01-14"}],
      ["set_staff_unavailable", {"staff_id": "ST-07", "date": "2027-01-13"}],
      ["remove_invigilator", {"exam_id": "EX-122", "staff_id": "ST-07"}],
      ["remove_invigilator", {"exam_id": "EX-123", "staff_id": "ST-07"}],
      ["remove_invigilator", {"exam_id": "EX-132", "staff_id": "ST-07"}],
      ["set_staff_unavailable", {"staff_id": "ST-07", "date": "2027-01-13"}],
      ["set_staff_unavailable", {"staff_id": "ST-07", "date": "2027-01-14"}],
      ["assign_invigilator", {"exam_id": "EX-122", "staff_id": "ST-11", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-123", "staff_id": "ST-06", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-132", "staff_id": "ST-05", "role": "support"}]],
     [["remove_invigilator", {"exam_id": "EX-122", "staff_id": "ST-07"}],
      ["remove_invigilator", {"exam_id": "EX-123", "staff_id": "ST-07"}],
      ["remove_invigilator", {"exam_id": "EX-132", "staff_id": "ST-07"}],
      ["set_staff_unavailable", {"staff_id": "ST-07", "date": "2027-01-13"}],
      ["set_staff_unavailable", {"staff_id": "ST-07", "date": "2027-01-14"}],
      ["assign_invigilator", {"exam_id": "EX-122", "staff_id": "ST-05", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-123", "staff_id": "ST-05", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-132", "staff_id": "ST-05", "role": "support"}]],
     lambda s, i: _final(s, i, changes={
         "assignments": _gone(i, [("EX-122", "ST-07"), ("EX-123", "ST-07"), ("EX-132", "ST-07")]),
         "staff": {"ST-07": {"unavailable": ["2027-01-13", "2027-01-14", "2027-01-15"]}}},
         new={"assignments": [_asg("EX-122", "ST-11", "lead"), _asg("EX-123", "ST-06", "support"),
                              _asg("EX-132", "ST-05", "support")]}),
     ref_errors=[4], mentions=["Femi Okoro"])

_add("T35", "exams", _exams(),
     "The Sports Hall is closed on Thursday 14 January because the heating has failed. Move each exam booked there "
     "that day to the smallest other room that can seat all its candidates and is free at that time. If no single "
     "free room is big enough, split the exam: the original sitting goes to the largest free room, filled to its "
     "capacity, and the rest of the candidates form a new sitting in the smallest free room that can take them. "
     "Staff a new sitting with free non-teaching staff only: the lowest-id lead-qualified one as lead, then the "
     "lowest-id other one as support, up to what policy requires. Existing invigilator assignments stay as they are.",
     [["get_exam_policy", {}], ["list_exams", {"date": "2027-01-14"}], ["list_rooms", {}],
      ["set_exam_room", {"exam_id": "EX-131", "room_id": "RM-01"}],
      ["set_exam_room", {"exam_id": "EX-133", "room_id": "RM-04"}],
      ["split_exam", {"exam_id": "EX-133", "room_id": "RM-05", "candidates": 15}],
      ["set_exam_room", {"exam_id": "EX-133", "room_id": "RM-04"}],
      ["list_staff_schedule", {"date": "2027-01-14"}],
      ["assign_invigilator", {"exam_id": "EX-900", "staff_id": "ST-02", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-900", "staff_id": "ST-03", "role": "support"}]],
     [["set_exam_room", {"exam_id": "EX-131", "room_id": "RM-01"}],
      ["split_exam", {"exam_id": "EX-133", "room_id": "RM-04", "candidates": 60}],
      ["assign_invigilator", {"exam_id": "EX-900", "staff_id": "ST-02", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-900", "staff_id": "ST-03", "role": "support"}]],
     lambda s, i: _final(s, i, changes={"exams": {"EX-131": {"room": "RM-01"},
                                                  "EX-133": {"room": "RM-04", "candidates": 60}}},
                         new={"exams": [{"paper": "Biology Paper 2", "subject": "Biology", "date": "2027-01-14",
                                         "start": "13:30", "duration": 120, "room": "RM-05", "candidates": 15,
                                         "status": "scheduled", "sitting_of": "EX-133"}],
                              "assignments": [_asg("EX-900", "ST-02", "lead"), _asg("EX-900", "ST-03", "support")]}),
     ref_errors=[4], mentions=["Sports Hall"], seq=[[5, 7, 8]])

_add("T36", "exams", _exams(),
     "Before Monday 11 January, please check the invigilation of each of that day's exams against our policy and "
     "fix only what is actually wrong. A teacher who has been put on an exam in their own subject is replaced by "
     "an eligible member of staff with the fewest sessions that day (ties: lower staff id). If an exam has more than "
     "one lead, the lead who was assigned first stays lead and any other becomes support. If an exam has more "
     "invigilators than it needs, take off the most recently assigned support. Exams that are already right must "
     "not be touched.",
     [["get_exam_policy", {}], ["list_exams", {"date": "2027-01-11"}], ["get_exam", {"exam_id": "EX-101"}],
      ["get_exam", {"exam_id": "EX-102"}], ["get_exam", {"exam_id": "EX-103"}],
      ["list_staff_schedule", {"date": "2027-01-11"}],
      ["assign_invigilator", {"exam_id": "EX-101", "staff_id": "ST-06", "role": "support"}],
      ["remove_invigilator", {"exam_id": "EX-101", "staff_id": "ST-12"}],
      ["assign_invigilator", {"exam_id": "EX-101", "staff_id": "ST-06", "role": "support"}],
      ["set_invigilator_role", {"exam_id": "EX-102", "staff_id": "ST-01", "role": "support"}],
      ["remove_invigilator", {"exam_id": "EX-103", "staff_id": "ST-22"}]],
     [["remove_invigilator", {"exam_id": "EX-101", "staff_id": "ST-12"}],
      ["assign_invigilator", {"exam_id": "EX-101", "staff_id": "ST-06", "role": "support"}],
      ["set_invigilator_role", {"exam_id": "EX-102", "staff_id": "ST-02", "role": "support"}],
      ["remove_invigilator", {"exam_id": "EX-103", "staff_id": "ST-22"}],
      ["remove_invigilator", {"exam_id": "EX-104", "staff_id": "ST-13"}]],
     lambda s, i: _final(s, i, changes={"assignments": dict(
         list(_gone(i, [("EX-101", "ST-12"), ("EX-103", "ST-22")]).items()) +
         [(_as_id(i, "EX-102", "ST-01"), {"role": "support"})])},
         new={"assignments": [_asg("EX-101", "ST-06", "support")]}),
     ref_errors=[6])

_add("T37", "exams", _exams(),
     "The papers for Chemistry Paper 1 will arrive late, so the exam has to move from Wednesday 13 January to Friday "
     "15 January at the same start time - in the same room if it is free then, otherwise in the smallest room that "
     "can seat everyone and is free. Keep every current invigilator who is still able to do it on Friday, in the "
     "same role; anyone who can't is replaced by an eligible member of staff in the same role with the fewest "
     "sessions on Friday (ties: lower staff id). Nothing else should change.",
     [["get_exam_policy", {}], ["list_exams", {"subject": "Chemistry"}], ["get_exam", {"exam_id": "EX-121"}],
      ["list_exams", {"date": "2027-01-15"}], ["list_rooms", {}],
      ["list_staff_schedule", {"date": "2027-01-15"}],
      ["remove_invigilator", {"exam_id": "EX-121", "staff_id": "ST-16"}],
      ["remove_invigilator", {"exam_id": "EX-121", "staff_id": "ST-03"}],
      ["remove_invigilator", {"exam_id": "EX-121", "staff_id": "ST-19"}],
      ["remove_invigilator", {"exam_id": "EX-121", "staff_id": "ST-20"}],
      ["reschedule_exam", {"exam_id": "EX-121", "date": "2027-01-15", "start": "09:00", "room_id": "RM-02"}],
      ["assign_invigilator", {"exam_id": "EX-121", "staff_id": "ST-16", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-121", "staff_id": "ST-03", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-121", "staff_id": "ST-20", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-121", "staff_id": "ST-04", "role": "support"}]],
     [["remove_invigilator", {"exam_id": "EX-121", "staff_id": "ST-16"}],
      ["remove_invigilator", {"exam_id": "EX-121", "staff_id": "ST-03"}],
      ["remove_invigilator", {"exam_id": "EX-121", "staff_id": "ST-19"}],
      ["remove_invigilator", {"exam_id": "EX-121", "staff_id": "ST-20"}],
      ["reschedule_exam", {"exam_id": "EX-121", "date": "2027-01-15", "start": "09:00", "room_id": "RM-02"}],
      ["assign_invigilator", {"exam_id": "EX-121", "staff_id": "ST-16", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-121", "staff_id": "ST-03", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-121", "staff_id": "ST-20", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-121", "staff_id": "ST-22", "role": "support"}]],
     lambda s, i: _final(s, i, changes={
         "exams": {"EX-121": {"date": "2027-01-15", "start": "09:00", "room": "RM-02"}},
         "assignments": _gone(i, [("EX-121", "ST-16"), ("EX-121", "ST-03"), ("EX-121", "ST-19"), ("EX-121", "ST-20")])},
         new={"assignments": [_asg("EX-121", "ST-16", "lead"), _asg("EX-121", "ST-03", "support"),
                              _asg("EX-121", "ST-20", "support"), _asg("EX-121", "ST-04", "support")]}),
     mentions=["Chemistry Paper 1"], seq=[[6, 10], [7, 10], [8, 10], [9, 10], [10, 11], [10, 12], [10, 13], [10, 14]])

_add("T38", "exams", _exams(),
     "Biology Paper 1 on Tuesday 12 January has more candidates than its room can seat. Split off exactly the "
     "overflow into a new sitting in the smallest room that can take them and is open and free at that time. The "
     "original sitting then needs fewer invigilators: move its most recently assigned support invigilator across to "
     "the new sitting as support. Then complete the new sitting's staffing as policy requires, using the lowest-id "
     "free non-teaching member of staff who can do what is still needed. Nothing else should change.",
     [["get_exam_policy", {}], ["list_exams", {"date": "2027-01-12"}], ["get_exam", {"exam_id": "EX-112"}],
      ["list_rooms", {}], ["split_exam", {"exam_id": "EX-112", "room_id": "RM-06", "candidates": 20}],
      ["list_staff_schedule", {"date": "2027-01-12"}],
      ["assign_invigilator", {"exam_id": "EX-900", "staff_id": "ST-13", "role": "support"}],
      ["remove_invigilator", {"exam_id": "EX-112", "staff_id": "ST-13"}],
      ["assign_invigilator", {"exam_id": "EX-900", "staff_id": "ST-13", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-900", "staff_id": "ST-02", "role": "lead"}]],
     [["split_exam", {"exam_id": "EX-112", "room_id": "RM-07", "candidates": 20}],
      ["split_exam", {"exam_id": "EX-112", "room_id": "RM-08", "candidates": 20}],
      ["remove_invigilator", {"exam_id": "EX-112", "staff_id": "ST-13"}],
      ["assign_invigilator", {"exam_id": "EX-900", "staff_id": "ST-13", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-900", "staff_id": "ST-02", "role": "lead"}]],
     lambda s, i: _final(s, i, changes={"exams": {"EX-112": {"candidates": 120}},
                                        "assignments": _gone(i, [("EX-112", "ST-13")])},
                         new={"exams": [{"paper": "Biology Paper 1", "subject": "Biology", "date": "2027-01-12",
                                         "start": "09:00", "duration": 120, "room": "RM-06", "candidates": 20,
                                         "status": "scheduled", "sitting_of": "EX-112"}],
                              "assignments": [_asg("EX-900", "ST-13", "support"), _asg("EX-900", "ST-02", "lead")]}),
     ref_errors=[6], mentions=["Biology Paper 1"], seq=[[4, 8, 9], [4, 7]])

_add("T39", "exams", _exams(),
     "Rita Varga has asked to be off on Friday 15 January. For each Friday exam she is down for, check whether "
     "someone eligible can take her place in the same role (a qualified lead if she is the lead): the replacement "
     "would be whoever has the fewest sessions that day (ties: lower staff id). Take her off every exam where a "
     "replacement exists and put the replacement in. If she can be replaced everywhere, also record her as "
     "unavailable on Friday; if even one of her Friday exams can't be covered, she stays on that one and is not "
     "recorded as unavailable. Don't confuse her with Hugo Varga, and change nothing else.",
     [["get_exam_policy", {}], ["find_staff", {"query": "Varga"}], ["list_staff_schedule", {"date": "2027-01-15"}],
      ["get_exam", {"exam_id": "EX-141"}], ["get_exam", {"exam_id": "EX-142"}],
      ["find_staff", {"lead_qualified": True}],
      ["remove_invigilator", {"exam_id": "EX-141", "staff_id": "ST-21"}],
      ["assign_invigilator", {"exam_id": "EX-141", "staff_id": "ST-04", "role": "support"}]],
     [["remove_invigilator", {"exam_id": "EX-141", "staff_id": "ST-21"}],
      ["assign_invigilator", {"exam_id": "EX-141", "staff_id": "ST-04", "role": "support"}],
      ["remove_invigilator", {"exam_id": "EX-142", "staff_id": "ST-21"}],
      ["set_invigilator_role", {"exam_id": "EX-142", "staff_id": "ST-03", "role": "lead"}]],
     lambda s, i: _final(s, i, changes={"assignments": _gone(i, [("EX-141", "ST-21")])},
                         new={"assignments": [_asg("EX-141", "ST-04", "support")]}),
     mentions=["Rita Varga", "Hugo Varga"])

_add("T40", "exams", _exams(),
     "For every exam next week whose lead invigilator is a teacher, check its support invigilators: if one of them "
     "is a non-teaching member of staff who holds the lead qualification, swap the two roles, so that person becomes "
     "lead and the teacher becomes support. If none of an exam's supports qualifies, or its lead is not a teacher, "
     "leave that exam exactly as it is. Don't add or remove anyone, and don't touch anything else - not even "
     "problems you may notice on the way.",
     [["find_staff", {}], ["list_exams", {}], ["list_exams", {"page": 2}], ["list_exams", {"page": 3}],
      ["set_invigilator_role", {"exam_id": "EX-101", "staff_id": "ST-16", "role": "lead"}],
      ["set_invigilator_role", {"exam_id": "EX-101", "staff_id": "ST-14", "role": "support"}],
      ["set_invigilator_role", {"exam_id": "EX-101", "staff_id": "ST-16", "role": "lead"}],
      ["set_invigilator_role", {"exam_id": "EX-123", "staff_id": "ST-09", "role": "support"}],
      ["set_invigilator_role", {"exam_id": "EX-123", "staff_id": "ST-01", "role": "lead"}],
      ["set_invigilator_role", {"exam_id": "EX-134", "staff_id": "ST-11", "role": "support"}],
      ["set_invigilator_role", {"exam_id": "EX-134", "staff_id": "ST-01", "role": "lead"}]],
     [["set_invigilator_role", {"exam_id": "EX-101", "staff_id": "ST-14", "role": "support"}],
      ["set_invigilator_role", {"exam_id": "EX-101", "staff_id": "ST-16", "role": "lead"}],
      ["set_invigilator_role", {"exam_id": "EX-123", "staff_id": "ST-09", "role": "support"}],
      ["set_invigilator_role", {"exam_id": "EX-123", "staff_id": "ST-07", "role": "lead"}],
      ["set_invigilator_role", {"exam_id": "EX-134", "staff_id": "ST-11", "role": "support"}],
      ["set_invigilator_role", {"exam_id": "EX-134", "staff_id": "ST-01", "role": "lead"}]],
     lambda s, i: _final(s, i, changes={"assignments": dict(
         (_as_id(i, e, st), {"role": r}) for e, st, r in [
             ("EX-101", "ST-14", "support"), ("EX-101", "ST-16", "lead"), ("EX-123", "ST-09", "support"),
             ("EX-123", "ST-01", "lead"), ("EX-134", "ST-11", "support"), ("EX-134", "ST-01", "lead")])}),
     ref_errors=[4], seq=[[5, 6], [7, 8], [9, 10]])


ENVS = {"library": LibraryEnv, "assets": AssetsEnv, "kitchen": KitchenEnv, "moderation": ModerationEnv,
        "exams": ExamsEnv}



# --------------------------------------------------------------------------
# extension T41-T75 (7 more tasks per environment; per-task fixture variants where needed)
# --------------------------------------------------------------------------
def _copy_rec(title, loc, status, cond="good", dest="", hold="", borrower="", due=""):
    return {"title": title, "location": loc, "status": status, "condition": cond, "destination": dest,
            "hold": hold, "borrower": borrower, "due": due}


def _hold_rec(patron, title, pickup, status, placed, cp="", until=""):
    return {"patron": patron, "title": title, "pickup": pickup, "status": status, "placed": "2026-" + placed,
            "copy": cp, "ready_until": until}


_CANC = {"status": "cancelled", "copy": "", "ready_until": ""}

_add("T41", "library", _lib(),
     "Mia Okonkwo's family is moving abroad, so please cancel every hold she still has active - including the ones "
     "already ready for her or on their way to her. Each copy freed up this way should go straight to the next hold "
     "in line for its title under the queue rules, wherever that patron collects; a copy nobody is eligible for "
     "stays where it is. Mia Okonkwo-Reed is a different patron, and nothing else should change.",
     [["find_patrons", {"query": "Mia Okonkwo"}], ["list_holds", {"patron_id": "P-205"}],
      ["list_holds", {"patron_id": "P-205", "page": 2}],
      ["cancel_hold", {"hold_id": "HD-530"}], ["cancel_hold", {"hold_id": "HD-531"}],
      ["cancel_hold", {"hold_id": "HD-532"}], ["cancel_hold", {"hold_id": "HD-533"}],
      ["cancel_hold", {"hold_id": "HD-534"}], ["cancel_hold", {"hold_id": "HD-535"}],
      ["receive_copy", {"copy_id": "CP-1902"}], ["cancel_hold", {"hold_id": "HD-535"}],
      ["cancel_hold", {"hold_id": "HD-536"}], ["list_holds", {"title_id": "BK-190"}],
      ["fill_hold", {"hold_id": "HD-554", "copy_id": "CP-1902"}], ["list_holds", {"title_id": "BK-160"}],
      ["fill_hold", {"hold_id": "HD-580", "copy_id": "CP-1601"}]],
     [["cancel_hold", {"hold_id": "HD-530"}], ["cancel_hold", {"hold_id": "HD-531"}],
      ["cancel_hold", {"hold_id": "HD-532"}], ["cancel_hold", {"hold_id": "HD-533"}],
      ["cancel_hold", {"hold_id": "HD-534"}], ["cancel_hold", {"hold_id": "HD-536"}],
      ["fill_hold", {"hold_id": "HD-580", "copy_id": "CP-1601"}]],
     lambda s, i: _final(s, i, changes={
         "holds": dict([(h, _CANC) for h in ("HD-530", "HD-531", "HD-532", "HD-533", "HD-534", "HD-535", "HD-536")] +
                       [("HD-554", {"status": "in_transit", "copy": "CP-1902"}),
                        ("HD-580", {"status": "in_transit", "copy": "CP-1601"})]),
         "copies": {"CP-1902": {"location": "CEN", "destination": "NOR", "status": "in_transit", "hold": "HD-554"},
                    "CP-1601": {"destination": "RIV", "status": "in_transit", "hold": "HD-580"}}}),
     ref_errors=[8], mentions=["Mia Okonkwo", "Mia Okonkwo-Reed"], seq=[[9, 10, 13], [11, 15]])


def _lib42():
    st = _lib()
    st["copies"]["CP-1503"] = _copy_rec("BK-150", "RIV", "on_shelf")
    st["copies"]["CP-1213"] = _copy_rec("BK-121", "RIV", "on_shelf")
    return st


def _c42(s, i):
    opts = []
    for x, y in (("CP-1101", "CP-1107"), ("CP-1107", "CP-1101")):
        opts.append(({"holds": {"HD-501": {"status": "in_transit", "copy": x},
                                "HD-504": {"status": "ready", "copy": y, "ready_until": _RDY},
                                "HD-533": {"status": "in_transit", "copy": "CP-1403"},
                                "HD-551": {"status": "in_transit", "copy": "CP-1503"}},
                      "copies": {x: {"status": "in_transit", "destination": "NOR", "hold": "HD-501"},
                                 y: {"status": "on_hold_shelf", "hold": "HD-504"},
                                 "CP-1403": {"status": "in_transit", "destination": "CEN", "hold": "HD-533"},
                                 "CP-1503": {"status": "in_transit", "destination": "CEN", "hold": "HD-551"}}}, None))
    return _any(*opts)(s, i)


_add("T42", "library", _lib42(),
     "Riverside wants its open shelf working for the waiting lists. Every copy on the open shelf at Riverside that "
     "can be used should go to the next hold in line for its title under the queue rules, wherever that patron "
     "collects; if a title has more than one such copy, keep working down its queue. Copies that nobody is eligible "
     "for, and copies that can't be used, stay where they are. Copies at other branches are not part of this, and "
     "nothing else should change.",
     [["get_library_info", {}], ["list_copies", {"location": "RIV", "status": "on_shelf"}],
      ["list_holds", {"title_id": "BK-110"}], ["list_holds", {"title_id": "BK-110", "page": 2}],
      ["fill_hold", {"hold_id": "HD-501", "copy_id": "CP-1101"}],
      ["fill_hold", {"hold_id": "HD-504", "copy_id": "CP-1107"}],
      ["list_holds", {"title_id": "BK-140"}], ["fill_hold", {"hold_id": "HD-533", "copy_id": "CP-1403"}],
      ["list_holds", {"title_id": "BK-150"}], ["fill_hold", {"hold_id": "HD-551", "copy_id": "CP-1503"}],
      ["list_holds", {"title_id": "BK-131"}], ["list_holds", {"title_id": "BK-121"}]],
     [["fill_hold", {"hold_id": "HD-501", "copy_id": "CP-1101"}],
      ["fill_hold", {"hold_id": "HD-504", "copy_id": "CP-1107"}]],
     _c42, mentions=["Riverside"], seq=[[4, 5]])


def _lib43():
    st = _lib()
    h, c = st["holds"], st["copies"]
    h["HD-508"] = _hold_rec("P-215", "BK-111", "CEN", "waiting", "10-18")
    h["HD-509"] = _hold_rec("P-212", "BK-131", "RIV", "waiting", "10-01")
    h["HD-510"] = _hold_rec("P-211", "BK-111", "NOR", "suspended", "10-27")
    h["HD-511"] = _hold_rec("P-202", "BK-131", "NOR", "ready", "10-20", "CP-1312", "2026-11-08")
    h["HD-512"] = _hold_rec("P-202", "BK-130", "NOR", "waiting", "09-28")
    h["HD-513"] = _hold_rec("P-221", "BK-111", "RIV", "ready", "10-25", "CP-1113", "2026-11-09")
    h["HD-514"] = _hold_rec("P-221", "BK-110", "RIV", "waiting", "10-03")
    h["HD-519"] = _hold_rec("P-219", "BK-111", "NOR", "waiting", "10-30")
    h["HD-522"] = _hold_rec("P-217", "BK-113", "NOR", "waiting", "10-30")
    c["CP-1312"] = _copy_rec("BK-131", "NOR", "on_hold_shelf", hold="HD-511")
    c["CP-1113"] = _copy_rec("BK-111", "RIV", "on_hold_shelf", hold="HD-513")
    return st


_add("T43", "library", _lib43(),
     "Some patrons have holds on both the regular book and the large-print edition of the same title, and they only "
     "need one. Wherever a patron has active holds on both editions, keep the one that is further along - ready "
     "first, then in transit, then waiting or suspended (those two count as equally far) - and if both are equally "
     "far along, keep the one placed earlier. Cancel the other one. Titles that merely look alike and other formats "
     "such as audiobooks don't count as the same book, and nothing else should change.",
     [["get_library_info", {}], ["list_holds", {"title_id": "BK-111"}], ["list_holds", {"title_id": "BK-131"}],
      ["list_holds", {"title_id": "BK-110"}], ["list_holds", {"title_id": "BK-110", "page": 2}],
      ["list_holds", {"title_id": "BK-130"}],
      ["cancel_hold", {"hold_id": "HD-506"}], ["cancel_hold", {"hold_id": "HD-509"}],
      ["cancel_hold", {"hold_id": "HD-507"}], ["cancel_hold", {"hold_id": "HD-512"}],
      ["cancel_hold", {"hold_id": "HD-514"}]],
     [["cancel_hold", {"hold_id": "HD-506"}], ["cancel_hold", {"hold_id": "HD-509"}],
      ["cancel_hold", {"hold_id": "HD-510"}], ["cancel_hold", {"hold_id": "HD-512"}],
      ["cancel_hold", {"hold_id": "HD-514"}]],
     lambda s, i: _final(s, i, changes={"holds": dict(
         (h, _CANC) for h in ("HD-506", "HD-509", "HD-507", "HD-512", "HD-514"))}))


def _lib44():
    st = _lib()
    st["copies"]["CP-1902"]["condition"] = "damaged"
    st["copies"]["CP-1301"]["condition"] = "damaged"
    return st


_add("T44", "library", _lib44(),
     "Two copies on their way between branches are known to be water-damaged: CP-1902 and CP-1301. Check each one "
     "in at the branch it is travelling to and then withdraw it. Each was meant for a patron's hold: cancel that "
     "hold and place a fresh hold for the same patron on the same title, with the same pickup branch (they have "
     "agreed to join the back of the queue). If the rules don't allow a patron another hold right now, leave them "
     "without one - don't cancel any of their other holds to make room. Nothing else should change.",
     [["get_library_info", {}], ["get_copy", {"copy_id": "CP-1902"}], ["get_copy", {"copy_id": "CP-1301"}],
      ["receive_copy", {"copy_id": "CP-1902"}], ["cancel_hold", {"hold_id": "HD-535"}],
      ["set_copy_status", {"copy_id": "CP-1902", "status": "withdrawn"}],
      ["place_hold", {"patron_id": "P-205", "title_id": "BK-190", "pickup_branch": "CEN"}],
      ["receive_copy", {"copy_id": "CP-1301"}], ["cancel_hold", {"hold_id": "HD-540"}],
      ["set_copy_status", {"copy_id": "CP-1301", "status": "withdrawn"}],
      ["place_hold", {"patron_id": "P-212", "title_id": "BK-130", "pickup_branch": "RIV"}]],
     [["receive_copy", {"copy_id": "CP-1902"}], ["cancel_hold", {"hold_id": "HD-535"}],
      ["set_copy_status", {"copy_id": "CP-1902", "status": "missing"}],
      ["receive_copy", {"copy_id": "CP-1301"}], ["cancel_hold", {"hold_id": "HD-540"}],
      ["set_copy_status", {"copy_id": "CP-1301", "status": "missing"}],
      ["place_hold", {"patron_id": "P-212", "title_id": "BK-130", "pickup_branch": "RIV"}]],
     lambda s, i: _final(s, i, changes={
         "holds": {"HD-535": _CANC, "HD-540": _CANC},
         "copies": {"CP-1902": {"location": "CEN", "destination": "", "status": "withdrawn", "hold": ""},
                    "CP-1301": {"location": "RIV", "destination": "", "status": "withdrawn", "hold": ""}}},
         new={"holds": [{"patron": "P-212", "title": "BK-130", "pickup": "RIV", "status": "waiting",
                         "placed": "2026-11-04", "copy": "", "ready_until": ""}]}),
     ref_errors=[6], seq=[[7, 8, 10]])


def _lib45():
    st = _lib()
    st["holds"]["HD-527"] = _hold_rec("P-207", "BK-121", "NOR", "suspended", "10-01")
    st["holds"]["HD-528"] = _hold_rec("P-207", "BK-170", "HIL", "suspended", "09-05")
    st["copies"]["CP-1504"] = _copy_rec("BK-150", "NOR", "on_shelf")
    return st


_add("T45", "library", _lib45(),
     "Felix Brandauer is back from a long trip. Reactivate every hold of his that is suspended. Where one of those "
     "holds is to be collected at a branch that is closed, switch it to his home branch, or to Central if his home "
     "branch is closed as well. Then, wherever the queue rules already allow a copy that is on the open shelf right "
     "now to go to one of his holds, fill it, using a copy at his pickup branch if there is one. Nobody else's holds "
     "should be filled or changed.",
     [["find_patrons", {"query": "Felix"}], ["list_holds", {"patron_id": "P-207"}], ["get_library_info", {}],
      ["update_hold", {"hold_id": "HD-520", "pickup_branch": "CEN"}],
      ["set_hold_suspended", {"hold_id": "HD-520", "suspended": False}],
      ["update_hold", {"hold_id": "HD-520", "pickup_branch": "CEN"}],
      ["set_hold_suspended", {"hold_id": "HD-528", "suspended": False}],
      ["update_hold", {"hold_id": "HD-528", "pickup_branch": "CEN"}],
      ["set_hold_suspended", {"hold_id": "HD-527", "suspended": False}],
      ["list_copies", {"title_id": "BK-121"}], ["fill_hold", {"hold_id": "HD-527", "copy_id": "CP-1211"}],
      ["list_copies", {"title_id": "BK-150"}], ["list_holds", {"title_id": "BK-150"}],
      ["list_copies", {"title_id": "BK-170"}]],
     [["set_hold_suspended", {"hold_id": "HD-520", "suspended": False}],
      ["update_hold", {"hold_id": "HD-520", "pickup_branch": "CEN"}],
      ["set_hold_suspended", {"hold_id": "HD-528", "suspended": False}],
      ["update_hold", {"hold_id": "HD-528", "pickup_branch": "CEN"}],
      ["set_hold_suspended", {"hold_id": "HD-527", "suspended": False}],
      ["fill_hold", {"hold_id": "HD-527", "copy_id": "CP-1211"}],
      ["fill_hold", {"hold_id": "HD-551", "copy_id": "CP-1504"}]],
     lambda s, i: _final(s, i, changes={
         "holds": {"HD-520": {"status": "waiting", "pickup": "CEN"}, "HD-528": {"status": "waiting", "pickup": "CEN"},
                   "HD-527": {"status": "ready", "copy": "CP-1211", "ready_until": _RDY}},
         "copies": {"CP-1211": {"status": "on_hold_shelf", "hold": "HD-527"}}}),
     ref_errors=[3], mentions=["Felix Brandauer", "Central"], seq=[[8, 10]])


def _lib46():
    st = _lib()
    st["copies"]["CP-1613"] = _copy_rec("BK-161", "CEN", "on_loan", borrower="P-211", due="2026-11-20")
    return st


def _c46(s, i):
    try:
        ids = [k for k, h in s["holds"].items() if k not in i["holds"] and h.get("title") == "BK-131"]
        if len(ids) != 1:
            return False
        bid = ids[0]
    except (KeyError, TypeError, AttributeError):
        return False
    return _final(s, i, changes={"copies": {"CP-1311": {"status": "in_transit", "destination": "CEN", "hold": bid}}},
                  new={"holds": [
                      {"patron": "P-211", "title": "BK-130", "pickup": "CEN", "status": "waiting", "placed": "2026-11-04",
                       "copy": "", "ready_until": ""},
                      {"patron": "P-211", "title": "BK-131", "pickup": "CEN", "status": "in_transit",
                       "placed": "2026-11-04", "copy": "CP-1311", "ready_until": ""},
                      {"patron": "P-211", "title": "BK-160", "pickup": "CEN", "status": "waiting", "placed": "2026-11-04",
                       "copy": "", "ready_until": ""}]})


_add("T46", "library", _lib46(),
     "Owen Price on our staff would like a hold, for collection at his home branch, on every title by Senna Holt or "
     "by Lydia Crane, in any format except audiobook. Skip any title he already has an active hold on or has out on "
     "loan. Then, for each new hold, if the queue rules already allow a copy that is on the open shelf somewhere to "
     "go to it, fill it with such a copy - one at his home branch if possible. Nothing else should change.",
     [["find_patrons", {"query": "Owen Price"}], ["get_patron", {"patron_id": "P-211"}],
      ["search_titles", {"query": "Senna Holt"}], ["search_titles", {"query": "Lydia Crane"}],
      ["list_holds", {"patron_id": "P-211"}],
      ["place_hold", {"patron_id": "P-211", "title_id": "BK-130", "pickup_branch": "CEN"}],
      ["place_hold", {"patron_id": "P-211", "title_id": "BK-131", "pickup_branch": "CEN"}],
      ["place_hold", {"patron_id": "P-211", "title_id": "BK-160", "pickup_branch": "CEN"}],
      ["list_copies", {"title_id": "BK-131"}], ["fill_hold", {"hold_id": "HD-601", "copy_id": "CP-1311"}],
      ["list_copies", {"title_id": "BK-130"}], ["list_copies", {"title_id": "BK-160"}]],
     [["place_hold", {"patron_id": "P-211", "title_id": "BK-130", "pickup_branch": "CEN"}],
      ["place_hold", {"patron_id": "P-211", "title_id": "BK-131", "pickup_branch": "CEN"}],
      ["place_hold", {"patron_id": "P-211", "title_id": "BK-160", "pickup_branch": "CEN"}],
      ["place_hold", {"patron_id": "P-211", "title_id": "BK-161", "pickup_branch": "CEN"}]],
     _c46, mentions=["Owen Price", "Senna Holt", "Lydia Crane"], seq=[[5, 6, 7, 9]])

_add("T47", "library", _lib(),
     "Two patrons can't collect books that are waiting for them. Grace Liang would rather pick up 'Winter Count' at "
     "Northgate, and Hannah Cole can't get to Hillcrest while it is closed, so she wants 'A Field Guide to Fungi' at "
     "Central instead. Do whatever the rules allow to get each of them a hold at the branch they asked for; if a "
     "ready hold can't simply be redirected, cancel it and place a new hold for them there (they accept going to the "
     "back of the queue). A copy freed this way goes to the next hold in line under the queue rules, if it can be "
     "sent anywhere. Nothing else should change.",
     [["get_library_info", {}], ["find_patrons", {"query": "Grace Liang"}], ["list_holds", {"patron_id": "P-208"}],
      ["update_hold", {"hold_id": "HD-594", "pickup_branch": "NOR"}], ["cancel_hold", {"hold_id": "HD-594"}],
      ["list_holds", {"title_id": "BK-180"}], ["fill_hold", {"hold_id": "HD-532", "copy_id": "CP-1801"}],
      ["place_hold", {"patron_id": "P-208", "title_id": "BK-180", "pickup_branch": "NOR"}],
      ["find_patrons", {"query": "Hannah Cole"}], ["list_holds", {"patron_id": "P-213"}],
      ["cancel_hold", {"hold_id": "HD-523"}],
      ["place_hold", {"patron_id": "P-213", "title_id": "BK-140", "pickup_branch": "CEN"}],
      ["list_holds", {"title_id": "BK-140"}]],
     [["cancel_hold", {"hold_id": "HD-594"}],
      ["place_hold", {"patron_id": "P-208", "title_id": "BK-180", "pickup_branch": "NOR"}],
      ["cancel_hold", {"hold_id": "HD-523"}],
      ["place_hold", {"patron_id": "P-213", "title_id": "BK-140", "pickup_branch": "CEN"}]],
     lambda s, i: _final(s, i, changes={
         "holds": {"HD-594": _CANC, "HD-523": _CANC,
                   "HD-532": {"status": "ready", "copy": "CP-1801", "ready_until": _RDY}},
         "copies": {"CP-1801": {"status": "on_hold_shelf", "hold": "HD-532"},
                    "CP-1402": {"status": "on_shelf", "hold": ""}}},
         new={"holds": [{"patron": "P-208", "title": "BK-180", "pickup": "NOR", "status": "waiting",
                         "placed": "2026-11-04", "copy": "", "ready_until": ""},
                        {"patron": "P-213", "title": "BK-140", "pickup": "CEN", "status": "waiting",
                         "placed": "2026-11-04", "copy": "", "ready_until": ""}]}),
     ref_errors=[3], mentions=["Grace Liang", "Hannah Cole", "Northgate", "Hillcrest", "Central"],
     seq=[[4, 7], [10, 11]])


# ---- assets T48-T54
_add("T48", "assets", _assets(),
     "Please clean up the Sales team's software. Take back every seat held by someone in Sales that has been "
     "inactive for longer than our policy allows. Then, for each licence that lost at least one seat this way, set "
     "its seat count to the number of seats still in use plus one spare; if that already is its seat count, leave "
     "the count alone. Licences that lost nobody keep their seat counts, and seats of other teams are not touched.",
     [["get_it_policy", {}], ["find_employees", {"dept": "Sales"}], ["get_employee", {"employee_id": "E-306"}],
      ["get_employee", {"employee_id": "E-313"}], ["get_employee", {"employee_id": "E-325"}],
      ["get_employee", {"employee_id": "E-326"}],
      ["revoke_seat", {"licence_id": "SW-22", "employee_id": "E-306"}],
      ["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-325"}],
      ["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-325"}],
      ["revoke_seat", {"licence_id": "SW-41", "employee_id": "E-325"}], ["list_licences", {}],
      ["set_seat_count", {"licence_id": "SW-11", "seats": 12}],
      ["set_seat_count", {"licence_id": "SW-21", "seats": 9}],
      ["set_seat_count", {"licence_id": "SW-22", "seats": 11}]],
     [["revoke_seat", {"licence_id": "SW-22", "employee_id": "E-306"}],
      ["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-325"}],
      ["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-325"}],
      ["revoke_seat", {"licence_id": "SW-41", "employee_id": "E-325"}],
      ["set_seat_count", {"licence_id": "SW-11", "seats": 11}],
      ["set_seat_count", {"licence_id": "SW-21", "seats": 8}],
      ["set_seat_count", {"licence_id": "SW-22", "seats": 10}],
      ["set_seat_count", {"licence_id": "SW-41", "seats": 7}]],
     lambda s, i: _final(s, i, changes={
         "seats": _revoked(i, [("SW-22", "E-306"), ("SW-11", "E-325"), ("SW-21", "E-325"), ("SW-41", "E-325")]),
         "licences": {"SW-11": {"seats": 12}, "SW-21": {"seats": 9}, "SW-22": {"seats": 11}}}),
     mentions=["Sales"])


def _dev_rec(kind, model, bought, status, who, enc=True, last=""):
    return {"kind": kind, "model": model, "purchased": bought, "status": status, "assigned_to": who,
            "encrypted": enc, "last_user": last}


def _assets49():
    st = _assets()
    d = st["devices"]
    d["D-4028"] = _dev_rec("laptop", "MacBook Pro 14", "2021-03-03", "in_use", "E-308")
    d["D-5038"] = _dev_rec("monitor", "Dell P2419 24-inch", "2020-01-01", "in_use", "E-302", False)
    d["D-5039"] = _dev_rec("monitor", "Dell U2723 27-inch", "2024-08-08", "in_use", "E-302", False)
    d["D-6013"] = _dev_rec("phone", "Pixel 7", "2023-01-01", "in_use", "E-326")
    d["D-6014"] = _dev_rec("phone", "iPhone 16", "2025-05-05", "in_use", "E-326")
    return st


_add("T49", "assets", _assets49(),
     "A few people hold more devices of one kind than our policy allows - left over from the old asset system. For "
     "each person who is over a limit, take back the extra devices of that kind, keeping the most recently "
     "purchased ones they have. Returned devices go into stock, except a laptop that is past the policy age limit, "
     "which is retired instead. People within their limits keep everything they have, and nothing else should "
     "change.",
     [["get_it_policy", {}], ["list_devices", {"kind": "laptop", "status": "in_use"}],
      ["list_devices", {"kind": "laptop", "status": "in_use", "page": 2}],
      ["list_devices", {"kind": "laptop", "status": "in_use", "page": 3}],
      ["list_devices", {"kind": "laptop", "status": "in_use", "page": 4}],
      ["list_devices", {"kind": "monitor", "status": "in_use"}],
      ["list_devices", {"kind": "monitor", "status": "in_use", "page": 2}],
      ["list_devices", {"kind": "phone", "status": "in_use"}],
      ["unassign_device", {"device_id": "D-4028", "destination": "in_stock"}], ["retire_device", {"device_id": "D-4028"}],
      ["unassign_device", {"device_id": "D-5038", "destination": "in_stock"}],
      ["unassign_device", {"device_id": "D-6013", "destination": "in_stock"}]],
     [["unassign_device", {"device_id": "D-4028", "destination": "in_stock"}],
      ["unassign_device", {"device_id": "D-5038", "destination": "in_stock"}],
      ["unassign_device", {"device_id": "D-6013", "destination": "in_stock"}]],
     lambda s, i: _final(s, i, changes={"devices": {
         "D-4028": {"status": "retired", "assigned_to": "", "last_user": "E-308"},
         "D-5038": {"status": "in_stock", "assigned_to": "", "last_user": "E-302"},
         "D-6013": {"status": "in_stock", "assigned_to": "", "last_user": "E-326"}}}))


def _assets50():
    st = _assets()
    st["employees"]["E-336"] = {"name": "Kasia Nowak", "dept": "Design", "status": "active", "end_date": ""}
    st["seats"]["SE-190"] = {"licence": "SW-12", "employee": "E-336", "assigned": "2025-03-03",
                             "last_used": "2026-11-10", "status": "active"}
    return st


_add("T50", "assets", _assets50(),
     "We're consolidating onto Figma Professional. Every Figma Organization seat holder who is allowed a Figma "
     "Professional seat should end up on Professional only: give them a Professional seat if they don't already have "
     "one, then take back their Organization seat. Anyone who has left the company just loses their Organization "
     "seat, and holders who aren't allowed Professional keep Organization. Finally set Figma Organization's seat count "
     "to exactly the number of its seats still in use. Nothing else should change.",
     [["get_it_policy", {}], ["list_licences", {}], ["list_seats", {"licence_id": "SW-12"}],
      ["get_employee", {"employee_id": "E-336"}], ["get_employee", {"employee_id": "E-303"}],
      ["assign_seat", {"licence_id": "SW-11", "employee_id": "E-336"}],
      ["revoke_seat", {"licence_id": "SW-12", "employee_id": "E-336"}],
      ["revoke_seat", {"licence_id": "SW-12", "employee_id": "E-303"}],
      ["revoke_seat", {"licence_id": "SW-12", "employee_id": "E-316"}],
      ["set_seat_count", {"licence_id": "SW-12", "seats": 1}]],
     [["revoke_seat", {"licence_id": "SW-12", "employee_id": "E-336"}],
      ["revoke_seat", {"licence_id": "SW-12", "employee_id": "E-303"}],
      ["revoke_seat", {"licence_id": "SW-12", "employee_id": "E-316"}],
      ["set_seat_count", {"licence_id": "SW-12", "seats": 1}]],
     lambda s, i: _final(s, i, changes={
         "seats": _revoked(i, [("SW-12", "E-336"), ("SW-12", "E-303"), ("SW-12", "E-316")]),
         "licences": {"SW-12": {"seats": 1}}},
         new={"seats": [_newseat("SW-11", "E-336")]}),
     mentions=["Figma Professional", "Figma Organization"], seq=[[5, 6]])

_add("T51", "assets", _assets(),
     "Four people asked for an extra monitor, in this order: Peter Holm, Ana Ferreira, Leon Weiss and Sven Aalto. "
     "Go through them in that order and give each one who is allowed another monitor under our policy one from "
     "stock: hand out the 24-inch monitors first, newest first, and once those have run out the 27-inch ones, newest "
     "first. Anyone already at the limit gets nothing - no swaps. Ana Ferreira-Lopes did not ask, and nothing else "
     "should change.",
     [["get_it_policy", {}], ["find_employees", {"query": "Peter Holm"}], ["get_employee", {"employee_id": "E-302"}],
      ["find_employees", {"query": "Ana Ferreira"}], ["get_employee", {"employee_id": "E-305"}],
      ["find_employees", {"query": "Leon Weiss"}], ["get_employee", {"employee_id": "E-330"}],
      ["find_employees", {"query": "Sven Aalto"}], ["get_employee", {"employee_id": "E-320"}],
      ["list_devices", {"kind": "monitor", "status": "in_stock"}],
      ["assign_device", {"device_id": "D-5034", "employee_id": "E-302"}],
      ["assign_device", {"device_id": "D-5032", "employee_id": "E-330"}],
      ["assign_device", {"device_id": "D-5035", "employee_id": "E-320"}]],
     [["assign_device", {"device_id": "D-5032", "employee_id": "E-302"}],
      ["assign_device", {"device_id": "D-5035", "employee_id": "E-330"}],
      ["assign_device", {"device_id": "D-5030", "employee_id": "E-320"}]],
     lambda s, i: _final(s, i, changes={"devices": {
         "D-5034": {"status": "in_use", "assigned_to": "E-302"}, "D-5032": {"status": "in_use", "assigned_to": "E-330"},
         "D-5035": {"status": "in_use", "assigned_to": "E-320"}}}),
     mentions=["Peter Holm", "Ana Ferreira", "Leon Weiss", "Sven Aalto", "Ana Ferreira-Lopes"])


def _assets52():
    st = _assets()
    for k, x in st["seats"].items():
        if x["licence"] == "SW-21" and x["employee"] == "E-310":
            x["last_used"] = "2026-08-30"
        if x["licence"] == "SW-21" and x["employee"] == "E-327":
            x["last_used"] = "2026-09-01"
    st["seats"]["SE-191"] = {"licence": "SW-22", "employee": "E-310", "assigned": "2025-06-06",
                             "last_used": "2026-11-01", "status": "active"}
    return st


_add("T52", "assets", _assets52(),
     "Downgrade idle Adobe Acrobat Pro users. Everyone whose Acrobat Pro seat has been inactive for longer than our "
     "policy allows moves to Adobe Acrobat Standard: take back the Pro seat and give them a Standard seat, unless "
     "they already have one. People who have left the company just lose the Pro seat. If Standard doesn't have "
     "enough free seats for this, raise its seat count by exactly the shortfall. Finally set Acrobat Pro's seat count "
     "to the number of Pro seats still in use. Nothing else should change.",
     [["get_it_policy", {}], ["list_licences", {}], ["list_seats", {"licence_id": "SW-21"}],
      ["list_seats", {"licence_id": "SW-21", "page": 2}], ["list_seats", {"licence_id": "SW-22"}],
      ["list_seats", {"licence_id": "SW-22", "page": 2}],
      ["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-310"}],
      ["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-316"}],
      ["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-325"}],
      ["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-327"}],
      ["assign_seat", {"licence_id": "SW-22", "employee_id": "E-325"}],
      ["set_seat_count", {"licence_id": "SW-22", "seats": 14}],
      ["assign_seat", {"licence_id": "SW-22", "employee_id": "E-325"}],
      ["assign_seat", {"licence_id": "SW-22", "employee_id": "E-327"}],
      ["set_seat_count", {"licence_id": "SW-21", "seats": 5}]],
     [["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-310"}],
      ["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-316"}],
      ["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-325"}],
      ["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-327"}],
      ["set_seat_count", {"licence_id": "SW-22", "seats": 15}],
      ["assign_seat", {"licence_id": "SW-22", "employee_id": "E-325"}],
      ["assign_seat", {"licence_id": "SW-22", "employee_id": "E-327"}],
      ["set_seat_count", {"licence_id": "SW-21", "seats": 5}]],
     lambda s, i: _final(s, i, changes={
         "seats": _revoked(i, [("SW-21", "E-310"), ("SW-21", "E-316"), ("SW-21", "E-325"), ("SW-21", "E-327")]),
         "licences": {"SW-22": {"seats": 14}, "SW-21": {"seats": 5}}},
         new={"seats": [_newseat("SW-22", "E-325"), _newseat("SW-22", "E-327")]}),
     ref_errors=[10], mentions=["Adobe Acrobat Pro", "Adobe Acrobat Standard"])

_add("T53", "assets", _assets(),
     "Laptop shuffle in Engineering. Rachel Stone's laptop is past our age limit and is to be retired. She takes over "
     "Dev Patel's current laptop, Dev takes over Kenji Mori's current laptop, and Kenji gets the newest MacBook in "
     "stock that policy allows us to hand out. Laptops that change hands are handed over directly - nothing goes to "
     "repair - and everyone else's devices stay exactly as they are.",
     [["get_it_policy", {}], ["find_employees", {"dept": "Engineering"}], ["get_employee", {"employee_id": "E-311"}],
      ["get_employee", {"employee_id": "E-315"}], ["get_employee", {"employee_id": "E-317"}],
      ["list_devices", {"kind": "laptop", "status": "in_stock"}],
      ["assign_device", {"device_id": "D-4014", "employee_id": "E-311"}],
      ["unassign_device", {"device_id": "D-4010", "destination": "in_stock"}], ["retire_device", {"device_id": "D-4010"}],
      ["unassign_device", {"device_id": "D-4014", "destination": "in_stock"}],
      ["assign_device", {"device_id": "D-4014", "employee_id": "E-311"}],
      ["unassign_device", {"device_id": "D-4015", "destination": "in_stock"}],
      ["assign_device", {"device_id": "D-4015", "employee_id": "E-315"}],
      ["assign_device", {"device_id": "D-4032", "employee_id": "E-317"}]],
     [["unassign_device", {"device_id": "D-4010", "destination": "in_stock"}], ["retire_device", {"device_id": "D-4010"}],
      ["unassign_device", {"device_id": "D-4014", "destination": "in_stock"}],
      ["assign_device", {"device_id": "D-4014", "employee_id": "E-311"}],
      ["unassign_device", {"device_id": "D-4015", "destination": "in_stock"}],
      ["assign_device", {"device_id": "D-4015", "employee_id": "E-315"}],
      ["assign_device", {"device_id": "D-4031", "employee_id": "E-317"}],
      ["assign_device", {"device_id": "D-4030", "employee_id": "E-317"}]],
     lambda s, i: _final(s, i, changes={"devices": {
         "D-4010": {"status": "retired", "assigned_to": "", "last_user": "E-311"},
         "D-4014": {"status": "in_use", "assigned_to": "E-311", "last_user": "E-315"},
         "D-4015": {"status": "in_use", "assigned_to": "E-315", "last_user": "E-317"},
         "D-4032": {"status": "in_use", "assigned_to": "E-317"}}}),
     ref_errors=[6], mentions=["Rachel Stone", "Dev Patel", "Kenji Mori", "MacBook"])


def _assets54():
    st = _assets()
    sx = st["seats"]
    sx["SE-192"] = {"licence": "SW-52", "employee": "E-309", "assigned": "2025-02-02", "last_used": "2026-10-01",
                    "status": "active"}
    sx["SE-193"] = {"licence": "SW-12", "employee": "E-309", "assigned": "2025-02-02", "last_used": "2026-09-20",
                    "status": "active"}
    sx["SE-194"] = {"licence": "SW-21", "employee": "E-309", "assigned": "2025-02-02", "last_used": "2026-10-18",
                    "status": "active"}
    st["licences"]["SW-52"]["seats"] = 11
    st["devices"]["D-5040"] = _dev_rec("monitor", "Dell P2419 24-inch", "2021-02-02", "in_use", "E-309", False)
    st["devices"]["D-5041"] = _dev_rec("monitor", "Dell U2723 27-inch", "2024-03-03", "in_use", "E-309", False)
    return st


_add("T54", "assets", _assets54(),
     "Jonas Lind leaves on 11 December. Today, only take back what he clearly no longer needs: every software seat "
     "he hasn't used within the last 30 days, and his older monitor - he keeps one monitor and his laptop until his "
     "last day. The monitor goes back into stock. Then, for each licence he gave a seat back on, if it now has more "
     "than two free seats, lower its seat count so that exactly two are free. Don't mark him as left yet, and nothing "
     "else should change.",
     [["get_it_policy", {}], ["find_employees", {"query": "Jonas Lind"}], ["get_employee", {"employee_id": "E-309"}],
      ["revoke_seat", {"licence_id": "SW-52", "employee_id": "E-309"}],
      ["revoke_seat", {"licence_id": "SW-12", "employee_id": "E-309"}],
      ["unassign_device", {"device_id": "D-5040", "destination": "in_stock"}],
      ["list_licences", {}], ["set_seat_count", {"licence_id": "SW-12", "seats": 5}]],
     [["revoke_seat", {"licence_id": "SW-52", "employee_id": "E-309"}],
      ["revoke_seat", {"licence_id": "SW-12", "employee_id": "E-309"}],
      ["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-309"}],
      ["unassign_device", {"device_id": "D-5041", "destination": "in_stock"}],
      ["set_seat_count", {"licence_id": "SW-12", "seats": 5}]],
     lambda s, i: _final(s, i, changes={
         "seats": _revoked(i, [("SW-52", "E-309"), ("SW-12", "E-309")]),
         "devices": {"D-5040": {"status": "in_stock", "assigned_to": "", "last_user": "E-309"}},
         "licences": {"SW-12": {"seats": 5}}}),
     mentions=["Jonas Lind"])


# ---- kitchen T55-T61
def _batch_rec(recipe, portions, date, line, status="planned", alloc=None):
    return {"recipe": recipe, "portions": portions, "date": "2026-" + date, "line": line, "status": status,
            "alloc": dict(alloc or {})}


def _kitchen55():
    st = _kitchen()
    st["batches"]["PB-806"] = _batch_rec("RC-13", 60, "11-26", "L2")
    return st


_add("T55", "kitchen", _kitchen55(),
     "Line 2 has broken down for Thursday 26 November. Move every batch still planned on Line 2 that day to another "
     "line, one at a time in order of batch id, each to whichever line it is allowed on that has the most room left "
     "at that point. If a batch doesn't fit anywhere in full, cut it to exactly the portions that fit on the allowed "
     "line with the most room and move it there; the portions that don't fit are dropped. Nothing else should "
     "change.",
     [["get_kitchen_info", {}], ["list_batches", {"date": "2026-11-26"}], ["get_recipe", {"recipe_id": "RC-12"}],
      ["get_recipe", {"recipe_id": "RC-13"}], ["get_recipe", {"recipe_id": "RC-03"}],
      ["update_batch", {"batch_id": "PB-805", "line": "L3"}], ["update_batch", {"batch_id": "PB-806", "line": "L1"}],
      ["update_batch", {"batch_id": "PB-851", "line": "L1"}],
      ["update_batch", {"batch_id": "PB-851", "portions": 40, "line": "L1"}]],
     [["update_batch", {"batch_id": "PB-851", "portions": 100, "line": "L1"}],
      ["update_batch", {"batch_id": "PB-805", "line": "L3"}], ["update_batch", {"batch_id": "PB-806", "line": "L3"}]],
     lambda s, i: _final(s, i, changes={"batches": {
         "PB-805": {"line": "L3"}, "PB-806": {"line": "L1"}, "PB-851": {"line": "L1", "portions": 40}}}),
     ref_errors=[7], mentions=["Line 2"], seq=[[5, 6, 8]])

_T56_ALLOC = {"LOT-0103": 6000, "LOT-0104": 18000, "LOT-0202": 20000, "LOT-0301": 4000, "LOT-0403": 18000}

_add("T56", "kitchen", _kitchen(),
     "Friday's Thai Green Curry batch (27 November) can't all be made from the stock we have. Cut it to the largest "
     "number of portions that usable stock not already allocated to other batches can fully cover, allocate every "
     "ingredient following the stock rules and "
     "release it to production. Then plan the portions you had to cut as a new batch of the same recipe on Monday "
     "30 November, on the same line, with nothing allocated yet. Nothing else should change.",
     [["get_kitchen_info", {}], ["list_batches", {"date": "2026-11-27"}], ["get_batch", {"batch_id": "PB-840"}],
      ["list_lots", {"ingredient_id": "ING-01"}], ["list_lots", {"ingredient_id": "ING-02"}],
      ["list_lots", {"ingredient_id": "ING-03"}], ["list_lots", {"ingredient_id": "ING-04"}],
      ["update_batch", {"batch_id": "PB-840", "portions": 200}],
      ["allocate_lot", {"batch_id": "PB-840", "lot_id": "LOT-0103", "grams": 6000}],
      ["allocate_lot", {"batch_id": "PB-840", "lot_id": "LOT-0104", "grams": 18000}],
      ["allocate_lot", {"batch_id": "PB-840", "lot_id": "LOT-0202", "grams": 20000}],
      ["allocate_lot", {"batch_id": "PB-840", "lot_id": "LOT-0301", "grams": 4000}],
      ["allocate_lot", {"batch_id": "PB-840", "lot_id": "LOT-0403", "grams": 18000}],
      ["release_batch", {"batch_id": "PB-840"}],
      ["create_batch", {"recipe_id": "RC-01", "portions": 50, "date": "2026-11-30", "line": "L1"}]],
     [["update_batch", {"batch_id": "PB-840", "portions": 216}],
      ["allocate_lot", {"batch_id": "PB-840", "lot_id": "LOT-0103", "grams": 6000}],
      ["allocate_lot", {"batch_id": "PB-840", "lot_id": "LOT-0104", "grams": 19920}],
      ["create_batch", {"recipe_id": "RC-01", "portions": 34, "date": "2026-11-30", "line": "L1"}]],
     lambda s, i: _final(s, i, changes={
         "batches": {"PB-840": {"portions": 200, "status": "released", "alloc": _T56_ALLOC}},
         "lots": _qty_after(i, _T56_ALLOC)},
         new={"batches": [{"recipe": "RC-01", "portions": 50, "date": "2026-11-30", "line": "L1", "status": "planned",
                           "alloc": {}}]}),
     mentions=["Thai Green Curry"], seq=[[7, 8, 9, 10, 11, 12, 13]])

_T57_ALLOC = {"LOT-0102": 12000, "LOT-1701": 6000, "LOT-0401": 5000, "LOT-0403": 4000, "LOT-1801": 3000}

_add("T57", "kitchen", _kitchen(),
     "A client needs 100 portions of Chicken Fajita Bowl made on Line 1 on Wednesday 25 November. Line 1 doesn't "
     "have room for it yet: make room by moving one planned batch off Line 1 that day - the smallest one whose move "
     "frees enough room - to whichever other line it is allowed on that has the most room. Then plan the client's "
     "batch and allocate all of its ingredients following the stock rules, but don't release it yet. Nothing else "
     "should change.",
     [["get_kitchen_info", {}], ["list_batches", {"date": "2026-11-25"}],
      ["update_batch", {"batch_id": "PB-830", "line": "L3"}], ["search_recipes", {"query": "Fajita"}],
      ["get_recipe", {"recipe_id": "RC-12"}],
      ["create_batch", {"recipe_id": "RC-12", "portions": 100, "date": "2026-11-25", "line": "L1"}],
      ["list_lots", {"ingredient_id": "ING-01"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0102", "grams": 12000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1701", "grams": 6000}],
      ["list_lots", {"ingredient_id": "ING-04"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0401", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0403", "grams": 4000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1801", "grams": 3000}]],
     [["update_batch", {"batch_id": "PB-830", "line": "L2"}],
      ["create_batch", {"recipe_id": "RC-12", "portions": 100, "date": "2026-11-25", "line": "L1"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0102", "grams": 12000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1701", "grams": 6000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0401", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0403", "grams": 4000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1801", "grams": 3000}]],
     lambda s, i: _final(s, i, changes={"batches": {"PB-830": {"line": "L3"}}}, new={"batches": [
         {"recipe": "RC-12", "portions": 100, "date": "2026-11-25", "line": "L1", "status": "planned",
          "alloc": _T57_ALLOC}]}),
     mentions=["Chicken Fajita Bowl", "Line 1"], seq=[[2, 5, 7, 8, 10, 11, 12]])


def _kitchen58():
    st = _kitchen()
    st["batches"]["PB-807"] = _batch_rec("RC-13", 150, "11-26", "L2")
    st["batches"]["PB-845"] = _batch_rec("RC-11", 60, "11-27", "L1")
    st["batches"]["PB-851"]["alloc"] = {"LOT-0104": 9600}
    return st


_add("T58", "kitchen", _kitchen58(),
     "Some lines are booked beyond their daily capacity this week (old data from before the checks existed). For "
     "every line and day where the planned and released batches add up to more than the line's capacity, cut the "
     "planned batch with the highest batch id on that line and day by exactly the excess. If that batch already has "
     "more of an ingredient allocated than its new size needs, take that allocation off and allocate exactly the new "
     "need again, following the stock rules. Nothing else should change.",
     [["get_kitchen_info", {}], ["list_batches", {"date": "2026-11-23"}], ["list_batches", {"date": "2026-11-24"}],
      ["list_batches", {"date": "2026-11-25"}], ["list_batches", {"date": "2026-11-26"}],
      ["list_batches", {"date": "2026-11-27"}], ["get_batch", {"batch_id": "PB-851"}],
      ["update_batch", {"batch_id": "PB-851", "portions": 70}],
      ["deallocate_lot", {"batch_id": "PB-851", "lot_id": "LOT-0104"}],
      ["update_batch", {"batch_id": "PB-851", "portions": 70}],
      ["allocate_lot", {"batch_id": "PB-851", "lot_id": "LOT-0103", "grams": 6000}],
      ["allocate_lot", {"batch_id": "PB-851", "lot_id": "LOT-0104", "grams": 2400}],
      ["update_batch", {"batch_id": "PB-845", "portions": 30}]],
     [["update_batch", {"batch_id": "PB-807", "portions": 100}],
      ["update_batch", {"batch_id": "PB-845", "portions": 30}]],
     lambda s, i: _final(s, i, changes={"batches": {
         "PB-851": {"portions": 70, "alloc": {"LOT-0103": 6000, "LOT-0104": 2400}}, "PB-845": {"portions": 30}}}),
     ref_errors=[7], seq=[[8, 9, 10, 11]])

_add("T59", "kitchen", _kitchen(),
     "Chestnut mushroom lot LOT-1302 arrived damaged and must be quarantined with reason damaged. Every planned batch "
     "that had mushrooms allocated from it needs replacement mushrooms from other usable lots, following the stock "
     "rules; deal with the batches in order of batch id. If a batch can't be fully covered, reduce it to the largest "
     "number of portions the remaining usable mushrooms can cover, and make its other allocations match the new "
     "size exactly: take off any allocation that is now too big and allocate the exact new need again, following "
     "the stock rules. Nothing else should change.",
     [["list_lots", {"ingredient_id": "ING-13"}], ["get_batch", {"batch_id": "PB-830"}],
      ["get_batch", {"batch_id": "PB-832"}],
      ["deallocate_lot", {"batch_id": "PB-830", "lot_id": "LOT-1302"}],
      ["deallocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1302"}],
      ["quarantine_lot", {"lot_id": "LOT-1302", "reason": "damaged"}],
      ["allocate_lot", {"batch_id": "PB-830", "lot_id": "LOT-1303", "grams": 9600}],
      ["update_batch", {"batch_id": "PB-832", "portions": 30}],
      ["deallocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1201"}],
      ["deallocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1401"}],
      ["update_batch", {"batch_id": "PB-832", "portions": 30}],
      ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1303", "grams": 2400}],
      ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1201", "grams": 2700}],
      ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1401", "grams": 450}]],
     [["deallocate_lot", {"batch_id": "PB-830", "lot_id": "LOT-1302"}],
      ["deallocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1302"}],
      ["quarantine_lot", {"lot_id": "LOT-1302", "reason": "damaged"}],
      ["allocate_lot", {"batch_id": "PB-830", "lot_id": "LOT-1303", "grams": 9600}],
      ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1303", "grams": 2400}]],
     lambda s, i: _final(s, i, changes={
         "lots": {"LOT-1302": {"status": "quarantined", "reason": "damaged"}},
         "batches": {"PB-830": {"alloc": {"LOT-1201": 10800, "LOT-1303": 9600, "LOT-1401": 1800}},
                     "PB-832": {"portions": 30, "alloc": {"LOT-1201": 2700, "LOT-1303": 2400, "LOT-1401": 450}}}}),
     ref_errors=[7], seq=[[6, 11]])


def _kitchen60():
    st = _kitchen()
    b = st["batches"]
    b["PB-860"] = _batch_rec("RC-09", 100, "11-18", "L2", alloc={"LOT-1303": 8000})
    b["PB-861"] = _batch_rec("RC-16", 80, "11-19", "L1")
    b["PB-862"] = _batch_rec("RC-06", 120, "11-19", "L1", alloc={"LOT-0701": 18000})
    b["PB-863"] = _batch_rec("RC-08", 50, "11-23", "L2")
    return st


_add("T60", "kitchen", _kitchen60(),
     "Some batches dated before today were never released or cancelled. Cancel every batch that is still planned "
     "but dated before today, so that their stock comes free. For each one whose recipe is still active, plan a "
     "replacement on the same weekday next week - seven days after its original date - on the same line with the "
     "same portions and nothing allocated; if that line can't take it that day, use whichever line it is allowed on "
     "that has the most room instead. Nothing else should change.",
     [["get_kitchen_info", {}], ["list_batches", {"status": "planned"}],
      ["get_recipe", {"recipe_id": "RC-09"}], ["get_recipe", {"recipe_id": "RC-16"}],
      ["get_recipe", {"recipe_id": "RC-06"}],
      ["cancel_batch", {"batch_id": "PB-860"}], ["cancel_batch", {"batch_id": "PB-861"}],
      ["cancel_batch", {"batch_id": "PB-862"}],
      ["create_batch", {"recipe_id": "RC-09", "portions": 100, "date": "2026-11-25", "line": "L2"}],
      ["create_batch", {"recipe_id": "RC-06", "portions": 120, "date": "2026-11-26", "line": "L1"}],
      ["list_batches", {"date": "2026-11-26"}],
      ["create_batch", {"recipe_id": "RC-06", "portions": 120, "date": "2026-11-26", "line": "L3"}]],
     [["cancel_batch", {"batch_id": "PB-860"}], ["cancel_batch", {"batch_id": "PB-861"}],
      ["cancel_batch", {"batch_id": "PB-862"}], ["cancel_batch", {"batch_id": "PB-863"}],
      ["create_batch", {"recipe_id": "RC-09", "portions": 100, "date": "2026-11-25", "line": "L2"}],
      ["create_batch", {"recipe_id": "RC-06", "portions": 120, "date": "2026-11-26", "line": "L2"}]],
     lambda s, i: _final(s, i, changes={"batches": dict(
         (b, {"status": "cancelled", "alloc": {}}) for b in ("PB-860", "PB-861", "PB-862"))},
         new={"batches": [
             {"recipe": "RC-09", "portions": 100, "date": "2026-11-25", "line": "L2", "status": "planned", "alloc": {}},
             {"recipe": "RC-06", "portions": 120, "date": "2026-11-26", "line": "L3", "status": "planned", "alloc": {}}]}),
     ref_errors=[9])

_add("T61", "kitchen", _kitchen(),
     "A client has cut two orders: Wednesday's Hummus Bowl goes down to 75 portions and Friday's Falafel Wrap to 40. "
     "Resize both batches. Every ingredient that already has stock allocated must end up allocated for exactly the "
     "new size, following the stock rules - take allocations off and allocate again where needed - while "
     "ingredients with nothing allocated yet stay that way. Don't release anything, and nothing else should change.",
     [["get_kitchen_info", {}], ["list_batches", {"date": "2026-11-25"}], ["get_batch", {"batch_id": "PB-833"}],
      ["list_batches", {"date": "2026-11-27"}], ["get_batch", {"batch_id": "PB-843"}],
      ["update_batch", {"batch_id": "PB-833", "portions": 75}],
      ["deallocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0701"}],
      ["deallocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0601"}],
      ["update_batch", {"batch_id": "PB-833", "portions": 75}],
      ["allocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0701", "grams": 11250}],
      ["allocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0601", "grams": 1500}],
      ["deallocate_lot", {"batch_id": "PB-843", "lot_id": "LOT-0602"}],
      ["update_batch", {"batch_id": "PB-843", "portions": 40}],
      ["list_lots", {"ingredient_id": "ING-06"}],
      ["allocate_lot", {"batch_id": "PB-843", "lot_id": "LOT-0601", "grams": 600}]],
     [["deallocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0701"}],
      ["deallocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0601"}],
      ["update_batch", {"batch_id": "PB-833", "portions": 75}],
      ["allocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0701", "grams": 11250}],
      ["allocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0601", "grams": 1500}],
      ["deallocate_lot", {"batch_id": "PB-843", "lot_id": "LOT-0602"}],
      ["update_batch", {"batch_id": "PB-843", "portions": 40}],
      ["allocate_lot", {"batch_id": "PB-843", "lot_id": "LOT-0602", "grams": 600}]],
     lambda s, i: _final(s, i, changes={"batches": {
         "PB-833": {"portions": 75, "alloc": {"LOT-0701": 11250, "LOT-0601": 1500}},
         "PB-843": {"portions": 40, "alloc": {"LOT-0601": 600}}}}),
     ref_errors=[5], mentions=["Hummus Bowl", "Falafel Wrap"])


# ---- moderation T62-T68
def _post_rec(author, comm, created, status, title):
    return {"author": author, "community": comm, "created": "2026-" + created, "status": status, "title": title}


def _rep_rec(post, reporter, reason, created, status="open", assignee=""):
    return {"post": post, "reporter": reporter, "reason": reason, "created": "2026-" + created, "status": status,
            "assignee": assignee}


def _user_rec(handle, joined, strikes=(), status="active", until="", verified=False):
    return {"handle": handle, "joined": joined, "strike_posts": list(strikes), "status": status,
            "suspended_until": until, "verified": verified, "trusted_flagger": False}


def _mod62():
    st = _mod()
    st["posts"]["PO-1984"] = _post_rec("U-181", "c/futbol", "11-01", "removed", "amenaza")
    u = st["users"]["U-181"]
    u["strike_posts"], u["status"], u["suspended_until"] = ["PO-1981", "PO-1982", "PO-1984"], "suspended", "2026-12-04"
    return st


_add("T62", "moderation", _mod62(),
     "Strikes expire once the post they were given for is more than 90 days old (counted from the day the post was "
     "created). Remove every expired strike given for posts in c/local-news and c/futbol. Afterwards, anyone who is "
     "suspended and has dropped below the suspension threshold is reinstated, signed off by Hector (M-08). Strikes "
     "for posts in other communities are not part of this, and nothing else should change.",
     [["get_mod_policy", {}], ["list_posts", {"community": "c/local-news", "status": "removed"}],
      ["list_posts", {"community": "c/futbol", "status": "removed"}],
      ["get_user", {"user_id": "U-140"}], ["get_user", {"user_id": "U-124"}], ["get_user", {"user_id": "U-180"}],
      ["get_user", {"user_id": "U-181"}],
      ["remove_strike", {"user_id": "U-140", "post_id": "PO-1940"}],
      ["remove_strike", {"user_id": "U-140", "post_id": "PO-1941"}],
      ["remove_strike", {"user_id": "U-140", "post_id": "PO-1942"}],
      ["remove_strike", {"user_id": "U-140", "post_id": "PO-1943"}],
      ["remove_strike", {"user_id": "U-181", "post_id": "PO-1981"}],
      ["set_user_status", {"user_id": "U-181", "status": "active", "approved_by": "M-08"}]],
     [["remove_strike", {"user_id": "U-140", "post_id": "PO-1940"}],
      ["remove_strike", {"user_id": "U-140", "post_id": "PO-1941"}],
      ["remove_strike", {"user_id": "U-140", "post_id": "PO-1942"}],
      ["remove_strike", {"user_id": "U-181", "post_id": "PO-1981"}],
      ["set_user_status", {"user_id": "U-181", "status": "active", "approved_by": "M-08"}]],
     lambda s, i: _final(s, i, changes={"users": {
         "U-140": {"strike_posts": []},
         "U-181": {"strike_posts": ["PO-1982", "PO-1984"], "status": "active", "suspended_until": ""}}}),
     mentions=["c/local-news", "c/futbol", "Hector"], seq=[[11, 12]])


def _mod63():
    st = _mod()
    st["users"]["U-191"] = _user_rec("fresh_acc_1", "2026-11-28")
    st["users"]["U-192"] = _user_rec("fresh_acc_2", "2026-11-29")
    st["users"]["U-193"] = _user_rec("newbie_nm", "2026-11-27")
    st["posts"]["PO-2070"] = _post_rec("U-161", "c/local-news", "11-28", "hidden", "report on the council vote")
    st["posts"]["PO-2071"] = _post_rec("U-164", "c/local-news", "11-29", "hidden", "opinion on the new stadium")
    for rid, post, rep, reason, created in [
        ("RP-3110", "PO-2070", "U-191", "harassment", "11-30"), ("RP-3111", "PO-2070", "U-192", "harassment", "11-30"),
        ("RP-3112", "PO-2070", "U-193", "harassment", "12-01"), ("RP-3113", "PO-2070", "U-151", "harassment", "11-29"),
        ("RP-3114", "PO-2071", "U-191", "hate", "11-30"), ("RP-3115", "PO-2071", "U-192", "hate", "12-01"),
        ("RP-3116", "PO-2071", "U-152", "hate", "11-30"), ("RP-3117", "PO-2071", "U-153", "hate", "11-30")]:
        st["reports"][rid] = _rep_rec(post, rep, reason, created)
    return st


_add("T63", "moderation", _mod63(),
     "c/local-news has been brigaded: freshly created accounts have been mass-reporting posts to get them hidden. "
     "For every hidden post in c/local-news, dismiss each open report on it filed by an account that is less than "
     "7 days old. Then, if fewer than two open reports from older accounts remain on that post, make it visible "
     "again; otherwise it stays hidden for review. Visible posts, and reports from established accounts, are left "
     "alone.",
     [["list_posts", {"community": "c/local-news", "status": "hidden"}], ["get_post", {"post_id": "PO-2070"}],
      ["get_post", {"post_id": "PO-2071"}], ["get_user", {"user_id": "U-191"}], ["get_user", {"user_id": "U-192"}],
      ["get_user", {"user_id": "U-193"}], ["get_user", {"user_id": "U-151"}], ["get_user", {"user_id": "U-152"}],
      ["get_user", {"user_id": "U-153"}],
      ["resolve_report", {"report_id": "RP-3110", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3111", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3112", "outcome": "dismissed"}],
      ["set_post_status", {"post_id": "PO-2070", "status": "visible"}],
      ["resolve_report", {"report_id": "RP-3114", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3115", "outcome": "dismissed"}]],
     [["resolve_report", {"report_id": "RP-3110", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3111", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3112", "outcome": "dismissed"}],
      ["set_post_status", {"post_id": "PO-2070", "status": "visible"}],
      ["resolve_report", {"report_id": "RP-3114", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3115", "outcome": "dismissed"}],
      ["set_post_status", {"post_id": "PO-2071", "status": "visible"}]],
     lambda s, i: _final(s, i, changes={
         "reports": dict((r, {"status": "dismissed"}) for r in ("RP-3110", "RP-3111", "RP-3112", "RP-3114", "RP-3115")),
         "posts": {"PO-2070": {"status": "visible"}}}),
     mentions=["c/local-news"])


def _mod64():
    st = _mod()
    st["users"]["U-135"] = _user_rec("chef_verena", "2020-05-05", verified=True)
    st["posts"]["PO-2072"] = _post_rec("U-135", "c/kochen", "11-28", "visible", "Wintermenue")
    for rid, post, rep, reason, created in [
        ("RP-3100", "PO-2030", "U-157", "spam", "11-28"), ("RP-3101", "PO-2072", "U-158", "hate", "11-29"),
        ("RP-3102", "PO-2033", "U-159", "threat", "11-30"), ("RP-3103", "PO-2096", "U-150", "spam", "12-01")]:
        st["reports"][rid] = _rep_rec(post, rep, reason, created)
    return st


_add("T64", "moderation", _mod64(),
     "Please clear the unassigned queue for our non-English communities, c/kochen and c/futbol. Take their open "
     "reports that nobody holds yet, oldest first, and assign each one to the moderator who is allowed to handle it "
     "and has the fewest open reports at that moment (ties: lower id). Reports in English-language communities, and "
     "reports someone already holds, are not part of this; nothing else should change.",
     [["get_mod_policy", {}], ["list_reports", {"community": "c/kochen", "status": "open"}],
      ["list_reports", {"community": "c/futbol", "status": "open"}], ["list_moderators", {}],
      ["get_user", {"user_id": "U-183"}], ["get_user", {"user_id": "U-130"}], ["get_user", {"user_id": "U-135"}],
      ["assign_report", {"report_id": "RP-3085", "moderator_id": "M-08"}],
      ["assign_report", {"report_id": "RP-3100", "moderator_id": "M-07"}],
      ["assign_report", {"report_id": "RP-3101", "moderator_id": "M-05"}],
      ["assign_report", {"report_id": "RP-3102", "moderator_id": "M-08"}],
      ["assign_report", {"report_id": "RP-3103", "moderator_id": "M-08"}]],
     [["assign_report", {"report_id": "RP-3085", "moderator_id": "M-08"}],
      ["assign_report", {"report_id": "RP-3100", "moderator_id": "M-07"}],
      ["assign_report", {"report_id": "RP-3101", "moderator_id": "M-07"}],
      ["assign_report", {"report_id": "RP-3102", "moderator_id": "M-08"}],
      ["assign_report", {"report_id": "RP-3103", "moderator_id": "M-02"}]],
     lambda s, i: _final(s, i, changes={"reports": {
         "RP-3085": {"assignee": "M-08"}, "RP-3100": {"assignee": "M-07"}, "RP-3101": {"assignee": "M-05"},
         "RP-3102": {"assignee": "M-08"}, "RP-3103": {"assignee": "M-08"}}}),
     mentions=["c/kochen", "c/futbol"])


def _mod65():
    st = _mod()
    u = st["users"]["U-140"]
    u["status"], u["strike_posts"] = "banned", ["PO-1940", "PO-1941", "PO-1942", "PO-1943", "PO-2042"]
    st["posts"]["PO-2042"] = _post_rec("U-140", "c/local-news", "11-20", "removed", "slur")
    st["posts"]["PO-2043"] = _post_rec("U-140", "c/gardening", "11-25", "hidden", "rant in a seed thread")
    st["reports"]["RP-3118"] = _rep_rec("PO-2045", "U-140", "spam", "11-30")
    return st


_add("T65", "moderation", _mod65(),
     "@grimwald has been banned. Remove every one of his posts that is still visible or hidden, and resolve the open "
     "reports on those posts as actioned - no new strikes, the ban already covers it. Any open report that he "
     "himself filed against someone else is dismissed. His account itself needs no further change. @grimwaldo is a "
     "different user, and nothing else should change.",
     [["find_users", {"query": "grimwald"}], ["list_posts", {"author_id": "U-140"}],
      ["get_post", {"post_id": "PO-2040"}], ["get_post", {"post_id": "PO-2041"}], ["get_post", {"post_id": "PO-2043"}],
      ["set_post_status", {"post_id": "PO-2040", "status": "removed"}],
      ["set_post_status", {"post_id": "PO-2041", "status": "removed"}],
      ["set_post_status", {"post_id": "PO-2043", "status": "removed"}],
      ["resolve_report", {"report_id": "RP-3050", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3051", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3052", "outcome": "actioned"}],
      ["list_reports", {"reporter_id": "U-140", "status": "open"}],
      ["resolve_report", {"report_id": "RP-3118", "outcome": "dismissed"}]],
     [["set_post_status", {"post_id": "PO-2040", "status": "removed"}],
      ["set_post_status", {"post_id": "PO-2041", "status": "removed"}],
      ["resolve_report", {"report_id": "RP-3050", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3051", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3052", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3118", "outcome": "dismissed"}]],
     lambda s, i: _final(s, i, changes={
         "posts": dict((p, {"status": "removed"}) for p in ("PO-2040", "PO-2041", "PO-2043")),
         "reports": {"RP-3050": {"status": "actioned"}, "RP-3051": {"status": "actioned"},
                     "RP-3052": {"status": "actioned"}, "RP-3118": {"status": "dismissed"}}}),
     mentions=["grimwald", "grimwaldo"])

_add("T66", "moderation", _mod(),
     "We have withdrawn @tidy_hunter's trusted-flagger privileges after a run of bad reports. Dismiss every report of "
     "theirs that is still open, whatever its reason. Their spam report that was already actioned led to a post "
     "being removed and its author getting a strike for it: undo that - put the post back up and take back that "
     "strike - but leave the report itself as it is. Reports filed by other people are not part of this, and nothing "
     "else should change.",
     [["find_users", {"query": "tidy_hunter"}], ["list_reports", {"reporter_id": "U-170"}],
      ["get_post", {"post_id": "PO-2025"}], ["get_user", {"user_id": "U-124"}],
      ["resolve_report", {"report_id": "RP-3020", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3021", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3022", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3023", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3024", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3027", "outcome": "dismissed"}],
      ["set_post_status", {"post_id": "PO-2025", "status": "visible"}],
      ["remove_strike", {"user_id": "U-124", "post_id": "PO-2025"}]],
     [["resolve_report", {"report_id": "RP-3020", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3021", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3022", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3023", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3024", "outcome": "dismissed"}],
      ["set_post_status", {"post_id": "PO-2025", "status": "visible"}],
      ["remove_strike", {"user_id": "U-124", "post_id": "PO-2025"}]],
     lambda s, i: _final(s, i, changes={
         "reports": dict((r, {"status": "dismissed"}) for r in
                         ("RP-3020", "RP-3021", "RP-3022", "RP-3023", "RP-3024", "RP-3027")),
         "posts": {"PO-2025": {"status": "visible"}}, "users": {"U-124": {"strike_posts": []}}}),
     mentions=["tidy_hunter"])


def _mod67():
    st = _mod()
    st["users"]["U-125"]["strike_posts"] = ["PO-2073", "PO-2074"]
    st["posts"]["PO-2073"] = _post_rec("U-125", "c/gardening", "10-10", "removed", "link farm")
    st["posts"]["PO-2074"] = _post_rec("U-125", "c/gardening", "11-01", "removed", "link farm again")
    st["posts"]["PO-2099"] = _post_rec("U-125", "c/gardening", "11-26", "removed", "discount seeds link")
    st["reports"]["RP-3119"] = _rep_rec("PO-2099", "U-154", "spam", "11-27")
    return st


_add("T67", "moderation", _mod67(),
     "Some removed posts still have reports open against them. For every removed post that has open or escalated "
     "reports, resolve those reports as actioned. Make sure the post's author has a strike for that post - give one "
     "where it is missing - and if that takes the author to the suspension threshold, suspend them with Chen (M-03) "
     "signing off. Removed posts without open reports, and everything else, stay as they are.",
     [["get_mod_policy", {}], ["list_posts", {"status": "removed"}], ["list_posts", {"status": "removed", "page": 2}],
      ["list_posts", {"status": "removed", "page": 3}], ["get_post", {"post_id": "PO-2010"}],
      ["get_post", {"post_id": "PO-2011"}], ["get_post", {"post_id": "PO-2099"}],
      ["get_user", {"user_id": "U-110"}], ["get_user", {"user_id": "U-125"}],
      ["resolve_report", {"report_id": "RP-3010", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3011", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3013", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3119", "outcome": "actioned"}],
      ["add_strike", {"user_id": "U-125", "post_id": "PO-2099"}],
      ["set_user_status", {"user_id": "U-125", "status": "suspended", "approved_by": "M-03"}]],
     [["resolve_report", {"report_id": "RP-3010", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3011", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3013", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3119", "outcome": "actioned"}],
      ["add_strike", {"user_id": "U-125", "post_id": "PO-2099"}]],
     lambda s, i: _final(s, i, changes={
         "reports": dict((r, {"status": "actioned"}) for r in ("RP-3010", "RP-3011", "RP-3013", "RP-3119")),
         "users": {"U-125": {"strike_posts": ["PO-2073", "PO-2074", "PO-2099"], "status": "suspended",
                             "suspended_until": "2026-12-09"}}}),
     mentions=["Chen"], seq=[[13, 14]])

_add("T68", "moderation", _mod(),
     "Some moderators are carrying too much. Go through the moderators in order of id. For each one holding more than "
     "75% of their maximum in open or escalated reports, move their most recently filed open report (not an "
     "escalated one; ties go to the higher report id) to the moderator allowed to handle it who has the fewest open "
     "reports at that moment (ties: lower id), and repeat until they are at or below 75%. Escalated reports stay "
     "where they are, and nothing else should change.",
     [["get_mod_policy", {}], ["list_moderators", {}], ["list_reports", {"assignee": "M-02"}],
      ["assign_report", {"report_id": "RP-3067", "moderator_id": "M-08"}],
      ["list_reports", {"assignee": "M-04"}],
      ["assign_report", {"report_id": "RP-3033", "moderator_id": "M-08"}],
      ["list_reports", {"assignee": "M-09"}], ["list_moderators", {}],
      ["assign_report", {"report_id": "RP-3097", "moderator_id": "M-01"}]],
     [["assign_report", {"report_id": "RP-3067", "moderator_id": "M-08"}],
      ["assign_report", {"report_id": "RP-3033", "moderator_id": "M-08"}],
      ["assign_report", {"report_id": "RP-3096", "moderator_id": "M-01"}]],
     lambda s, i: _final(s, i, changes={"reports": {
         "RP-3067": {"assignee": "M-08"}, "RP-3033": {"assignee": "M-08"}, "RP-3097": {"assignee": "M-01"}}}))


# ---- exams T69-T75
_add("T69", "exams", _exams(),
     "Candidate numbers have dropped for two papers: Maths Paper 2 now has 70 candidates and English Language Paper "
     "1 has 80. Update both. Each of them should then have exactly the number of invigilators policy requires - take "
     "off the most recently assigned support invigilators first - and should sit in the smallest room that seats "
     "its candidates and is free at that time, which may well be the room it is already in. Nothing else should "
     "change.",
     [["get_exam_policy", {}], ["list_exams", {"subject": "Maths"}], ["list_exams", {"subject": "English"}],
      ["get_exam", {"exam_id": "EX-123"}], ["get_exam", {"exam_id": "EX-134"}], ["list_rooms", {}],
      ["set_exam_candidates", {"exam_id": "EX-123", "candidates": 70}],
      ["remove_invigilator", {"exam_id": "EX-123", "staff_id": "ST-22"}],
      ["remove_invigilator", {"exam_id": "EX-123", "staff_id": "ST-04"}],
      ["list_exams", {"date": "2027-01-13"}], ["set_exam_room", {"exam_id": "EX-123", "room_id": "RM-02"}],
      ["set_exam_candidates", {"exam_id": "EX-134", "candidates": 80}],
      ["remove_invigilator", {"exam_id": "EX-134", "staff_id": "ST-17"}],
      ["list_exams", {"date": "2027-01-14"}]],
     [["set_exam_candidates", {"exam_id": "EX-123", "candidates": 70}],
      ["remove_invigilator", {"exam_id": "EX-123", "staff_id": "ST-07"}],
      ["remove_invigilator", {"exam_id": "EX-123", "staff_id": "ST-01"}],
      ["set_exam_room", {"exam_id": "EX-123", "room_id": "RM-02"}],
      ["set_exam_candidates", {"exam_id": "EX-134", "candidates": 80}],
      ["remove_invigilator", {"exam_id": "EX-134", "staff_id": "ST-01"}]],
     lambda s, i: _final(s, i, changes={
         "exams": {"EX-123": {"candidates": 70, "room": "RM-02"}, "EX-134": {"candidates": 80}},
         "assignments": _gone(i, [("EX-123", "ST-22"), ("EX-123", "ST-04"), ("EX-134", "ST-17")])}),
     mentions=["Maths Paper 2", "English Language Paper 1"])

_add("T70", "exams", _exams(),
     "Please swap the start times of French Paper 1 and Art History Paper 1 on Wednesday 13 January: French moves to "
     "Art History's start time and Art History to French's, each keeping its own room. Keep each exam's current "
     "invigilators wherever they can still do the new time, in the same role. Put all of those back first; then "
     "replace anyone who can't stay with an eligible member of staff in the same role who has the fewest sessions "
     "that day at that point (ties: lower staff id). Nothing else should change.",
     [["get_exam_policy", {}], ["list_exams", {"date": "2027-01-13"}],
      ["list_staff_schedule", {"date": "2027-01-13"}],
      ["remove_invigilator", {"exam_id": "EX-122", "staff_id": "ST-07"}],
      ["remove_invigilator", {"exam_id": "EX-122", "staff_id": "ST-15"}],
      ["remove_invigilator", {"exam_id": "EX-124", "staff_id": "ST-02"}],
      ["remove_invigilator", {"exam_id": "EX-124", "staff_id": "ST-17"}],
      ["reschedule_exam", {"exam_id": "EX-122", "date": "2027-01-13", "start": "13:30", "room_id": "RM-04"}],
      ["reschedule_exam", {"exam_id": "EX-124", "date": "2027-01-13", "start": "09:00", "room_id": "RM-08"}],
      ["assign_invigilator", {"exam_id": "EX-124", "staff_id": "ST-02", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-124", "staff_id": "ST-17", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-122", "staff_id": "ST-11", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-122", "staff_id": "ST-15", "role": "support"}]],
     [["remove_invigilator", {"exam_id": "EX-122", "staff_id": "ST-07"}],
      ["remove_invigilator", {"exam_id": "EX-122", "staff_id": "ST-15"}],
      ["remove_invigilator", {"exam_id": "EX-124", "staff_id": "ST-02"}],
      ["remove_invigilator", {"exam_id": "EX-124", "staff_id": "ST-17"}],
      ["reschedule_exam", {"exam_id": "EX-122", "date": "2027-01-13", "start": "13:30", "room_id": "RM-04"}],
      ["reschedule_exam", {"exam_id": "EX-124", "date": "2027-01-13", "start": "09:00", "room_id": "RM-08"}],
      ["assign_invigilator", {"exam_id": "EX-124", "staff_id": "ST-02", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-124", "staff_id": "ST-17", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-122", "staff_id": "ST-14", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-122", "staff_id": "ST-15", "role": "support"}]],
     lambda s, i: _final(s, i, changes={
         "exams": {"EX-122": {"start": "13:30"}, "EX-124": {"start": "09:00"}},
         "assignments": _gone(i, [("EX-122", "ST-07"), ("EX-122", "ST-15"), ("EX-124", "ST-02"), ("EX-124", "ST-17")])},
         new={"assignments": [_asg("EX-124", "ST-02", "lead"), _asg("EX-124", "ST-17", "support"),
                              _asg("EX-122", "ST-11", "lead"), _asg("EX-122", "ST-15", "support")]}),
     mentions=["French Paper 1", "Art History Paper 1"],
     seq=[[3, 7], [4, 7], [5, 8], [6, 8], [7, 11], [7, 12], [8, 9], [8, 10]])


def _exams71():
    st = _exams()
    for sid, ex in (("ST-03", "EX-131"), ("ST-16", "EX-133")):
        st["staff"][sid]["unavailable"] = sorted(st["staff"][sid]["unavailable"] + ["2027-01-14"])
        for k, a in st["assignments"].items():
            if a["staff"] == sid and a["exam"] == ex:
                a["status"] = "removed"
    return st


_add("T71", "exams", _exams71(),
     "Two invigilators have dropped out of Thursday 14 January and their places have been cleared. Bring every "
     "Thursday exam back up to exactly the number of invigilators policy requires. Go through the exams in order "
     "of start time, then exam id; an exam without a lead gets a qualified lead first, then any supports it still "
     "needs. Each time, pick the eligible member of staff with the fewest sessions that day (ties: lower staff id). "
     "Nothing else should change.",
     [["get_exam_policy", {}], ["list_exams", {"date": "2027-01-14"}], ["get_exam", {"exam_id": "EX-131"}],
      ["get_exam", {"exam_id": "EX-133"}], ["list_staff_schedule", {"date": "2027-01-14"}],
      ["find_staff", {"lead_qualified": True}],
      ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-08", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-133", "staff_id": "ST-05", "role": "lead"}]],
     [["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-08", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-133", "staff_id": "ST-21", "role": "lead"}]],
     lambda s, i: _final(s, i, new={"assignments": [_asg("EX-131", "ST-08", "support"), _asg("EX-133", "ST-05", "lead")]}))

_add("T72", "exams", _exams(),
     "To free up the big halls, go through every exam on Thursday 14 and Friday 15 January in order of date, start "
     "time and exam id, and move each one to the smallest room that seats its candidates and is free at that time - "
     "but only if that room is smaller than the one it is in now. Look at each exam once, in that order, with the "
     "rooms as they are at that point. Invigilators stay as they are, and nothing else should change.",
     [["get_exam_policy", {}], ["list_rooms", {}], ["list_exams", {"date": "2027-01-14"}],
      ["list_exams", {"date": "2027-01-15"}],
      ["set_exam_room", {"exam_id": "EX-140", "room_id": "RM-08"}],
      ["set_exam_room", {"exam_id": "EX-142", "room_id": "RM-02"}],
      ["set_exam_room", {"exam_id": "EX-145", "room_id": "RM-07"}]],
     [["set_exam_room", {"exam_id": "EX-140", "room_id": "RM-08"}],
      ["set_exam_room", {"exam_id": "EX-142", "room_id": "RM-02"}],
      ["set_exam_room", {"exam_id": "EX-145", "room_id": "RM-07"}],
      ["set_exam_room", {"exam_id": "EX-143", "room_id": "RM-08"}]],
     lambda s, i: _final(s, i, changes={"exams": {
         "EX-140": {"room": "RM-08"}, "EX-142": {"room": "RM-02"}, "EX-145": {"room": "RM-07"}}}))

_add("T73", "exams", _exams(),
     "Victor Lam has far too many sessions next week; bring him down to at most 4. Take him off his sessions starting "
     "from the latest one (by date and start time). Each time, put in a replacement in the same role: the eligible "
     "member of staff, not already on that exam, with the fewest sessions that day (ties: lower staff id). If a "
     "session has no possible replacement, he stays on it and you move on to his next latest session. Nothing else "
     "should change.",
     [["get_exam_policy", {}], ["find_staff", {"query": "Victor Lam"}], ["find_staff", {"lead_qualified": True}],
      ["list_staff_schedule", {"date": "2027-01-15"}],
      ["remove_invigilator", {"exam_id": "EX-141", "staff_id": "ST-02"}],
      ["assign_invigilator", {"exam_id": "EX-141", "staff_id": "ST-16", "role": "lead"}],
      ["list_staff_schedule", {"date": "2027-01-14"}],
      ["remove_invigilator", {"exam_id": "EX-132", "staff_id": "ST-02"}],
      ["assign_invigilator", {"exam_id": "EX-132", "staff_id": "ST-05", "role": "lead"}]],
     [["remove_invigilator", {"exam_id": "EX-145", "staff_id": "ST-02"}],
      ["remove_invigilator", {"exam_id": "EX-141", "staff_id": "ST-02"}],
      ["assign_invigilator", {"exam_id": "EX-141", "staff_id": "ST-16", "role": "lead"}]],
     lambda s, i: _final(s, i, changes={
         "assignments": _gone(i, [("EX-141", "ST-02"), ("EX-132", "ST-02")])},
         new={"assignments": [_asg("EX-141", "ST-16", "lead"), _asg("EX-132", "ST-05", "lead")]}),
     mentions=["Victor Lam"], seq=[[4, 5], [7, 8]])

_add("T74", "exams", _exams(),
     "Nobody should invigilate more than 9 hours in total next week, counting each exam's full duration. For each "
     "member of staff over that limit, in order of staff id, take them off their longest exam - the latest one if "
     "several are equally long - and repeat until they are within the limit. Replace them each time in the same "
     "role with the eligible member of staff, not already on that exam and not the person just taken off, who has "
     "the fewest sessions that day (ties: lower staff id) and who would still be within the 9 hours afterwards. "
     "Nothing else should change.",
     [["get_exam_policy", {}], ["list_exams", {}], ["list_exams", {"page": 2}], ["list_exams", {"page": 3}],
      ["list_staff_schedule", {"date": "2027-01-15"}],
      ["remove_invigilator", {"exam_id": "EX-142", "staff_id": "ST-03"}],
      ["assign_invigilator", {"exam_id": "EX-142", "staff_id": "ST-06", "role": "support"}],
      ["list_staff_schedule", {"date": "2027-01-14"}],
      ["remove_invigilator", {"exam_id": "EX-133", "staff_id": "ST-16"}],
      ["assign_invigilator", {"exam_id": "EX-133", "staff_id": "ST-05", "role": "lead"}]],
     [["remove_invigilator", {"exam_id": "EX-142", "staff_id": "ST-03"}],
      ["assign_invigilator", {"exam_id": "EX-142", "staff_id": "ST-04", "role": "support"}],
      ["remove_invigilator", {"exam_id": "EX-133", "staff_id": "ST-16"}],
      ["assign_invigilator", {"exam_id": "EX-133", "staff_id": "ST-05", "role": "lead"}],
      ["remove_invigilator", {"exam_id": "EX-145", "staff_id": "ST-02"}]],
     lambda s, i: _final(s, i, changes={
         "assignments": _gone(i, [("EX-142", "ST-03"), ("EX-133", "ST-16")])},
         new={"assignments": [_asg("EX-142", "ST-06", "support"), _asg("EX-133", "ST-05", "lead")]}),
     seq=[[5, 6], [8, 9]])

_add("T75", "exams", _exams(),
     "Two pairs of exams on Friday 15 January should swap rooms: Music Paper 1 with German Paper 1 in the morning, "
     "and Physics Paper 3 with Spanish Paper 1 in the afternoon. When you are done, each of the four must be in its "
     "partner's original room, no other exam may have changed room or time, and all invigilators stay exactly as "
     "they are.",
     [["get_exam_policy", {}], ["list_exams", {"date": "2027-01-15"}], ["list_rooms", {}],
      ["set_exam_room", {"exam_id": "EX-140", "room_id": "RM-04"}],
      ["set_exam_room", {"exam_id": "EX-140", "room_id": "RM-02"}],
      ["set_exam_room", {"exam_id": "EX-141", "room_id": "RM-01"}],
      ["set_exam_room", {"exam_id": "EX-140", "room_id": "RM-04"}],
      ["set_exam_room", {"exam_id": "EX-143", "room_id": "RM-02"}],
      ["set_exam_room", {"exam_id": "EX-144", "room_id": "RM-03"}],
      ["set_exam_room", {"exam_id": "EX-143", "room_id": "RM-04"}]],
     [["set_exam_room", {"exam_id": "EX-140", "room_id": "RM-02"}],
      ["set_exam_room", {"exam_id": "EX-141", "room_id": "RM-01"}],
      ["set_exam_room", {"exam_id": "EX-140", "room_id": "RM-04"}]],
     lambda s, i: _final(s, i, changes={"exams": {
         "EX-140": {"room": "RM-04"}, "EX-141": {"room": "RM-01"}, "EX-143": {"room": "RM-04"},
         "EX-144": {"room": "RM-03"}}}),
     ref_errors=[3], mentions=["Music Paper 1", "German Paper 1", "Physics Paper 3", "Spanish Paper 1"],
     seq=[[4, 5, 6], [7, 8, 9]])



# --------------------------------------------------------------------------
# calibration tasks TP1-TP6 (same style and difficulty as the test tasks)
# --------------------------------------------------------------------------
_add("TP1", "library", _lib(),
     "Results of this morning's shelf check at Northgate. These copies could not be found: CP-1111, CP-1211, CP-1121 "
     "and CP-1104. Mark each of them missing - but only if the system says it should be sitting at Northgate; a copy "
     "that is out on loan isn't missing. If one of them was waiting on the hold shelf for somebody, cancel that hold "
     "first. Two copies that had been marked missing turned up: CP-1108 and CP-1404. Put them back into "
     "circulation: a found copy that is damaged is withdrawn instead, and a usable one goes straight to the next "
     "hold in line for its title, following the queue rules. Change nothing else.",
     [["get_library_info", {}], ["list_copies", {"location": "NOR"}], ["list_copies", {"location": "NOR", "page": 2}],
      ["set_copy_status", {"copy_id": "CP-1111", "status": "missing"}],
      ["set_copy_status", {"copy_id": "CP-1211", "status": "missing"}],
      ["set_copy_status", {"copy_id": "CP-1121", "status": "missing"}],
      ["cancel_hold", {"hold_id": "HD-596"}],
      ["set_copy_status", {"copy_id": "CP-1121", "status": "missing"}],
      ["set_copy_status", {"copy_id": "CP-1108", "status": "on_shelf"}],
      ["list_holds", {"title_id": "BK-110"}],
      ["fill_hold", {"hold_id": "HD-501", "copy_id": "CP-1108"}],
      ["set_copy_status", {"copy_id": "CP-1404", "status": "withdrawn"}]],
     [["set_copy_status", {"copy_id": "CP-1111", "status": "missing"}],
      ["set_copy_status", {"copy_id": "CP-1211", "status": "missing"}],
      ["set_copy_status", {"copy_id": "CP-1108", "status": "on_shelf"}],
      ["fill_hold", {"hold_id": "HD-501", "copy_id": "CP-1108"}],
      ["set_copy_status", {"copy_id": "CP-1404", "status": "on_shelf"}]],
     lambda s, i: _final(s, i, changes={
         "copies": {"CP-1111": {"status": "missing"}, "CP-1211": {"status": "missing"},
                    "CP-1121": {"status": "missing", "hold": ""},
                    "CP-1108": {"status": "on_hold_shelf", "hold": "HD-501"}, "CP-1404": {"status": "withdrawn"}},
         "holds": {"HD-596": {"status": "cancelled", "copy": "", "ready_until": ""},
                   "HD-501": {"status": "ready", "copy": "CP-1108", "ready_until": _RDY}}}),
     ref_errors=[5], mentions=["Northgate"], seq=[[8, 10]])

_add("TP2", "assets", _assets(),
     "Zendesk Agent seats are meant for the Support team only. Take back every Zendesk Agent seat held by someone "
     "outside Support or by anyone who has left the company. Then make sure every active member of the Support team "
     "has a seat; if there aren't enough free seats for that, raise the seat count by exactly the shortfall. Other "
     "licences and everything else stay as they are.",
     [["get_it_policy", {}], ["list_licences", {}], ["list_seats", {"licence_id": "SW-41"}],
      ["find_employees", {"dept": "Support"}],
      ["revoke_seat", {"licence_id": "SW-41", "employee_id": "E-313"}],
      ["revoke_seat", {"licence_id": "SW-41", "employee_id": "E-325"}],
      ["revoke_seat", {"licence_id": "SW-41", "employee_id": "E-335"}],
      ["assign_seat", {"licence_id": "SW-41", "employee_id": "E-314"}],
      ["assign_seat", {"licence_id": "SW-41", "employee_id": "E-324"}],
      ["assign_seat", {"licence_id": "SW-41", "employee_id": "E-331"}],
      ["assign_seat", {"licence_id": "SW-41", "employee_id": "E-333"}],
      ["set_seat_count", {"licence_id": "SW-41", "seats": 9}],
      ["assign_seat", {"licence_id": "SW-41", "employee_id": "E-333"}]],
     [["revoke_seat", {"licence_id": "SW-41", "employee_id": "E-313"}],
      ["revoke_seat", {"licence_id": "SW-41", "employee_id": "E-325"}],
      ["set_seat_count", {"licence_id": "SW-41", "seats": 10}],
      ["assign_seat", {"licence_id": "SW-41", "employee_id": "E-314"}],
      ["assign_seat", {"licence_id": "SW-41", "employee_id": "E-324"}],
      ["assign_seat", {"licence_id": "SW-41", "employee_id": "E-331"}],
      ["assign_seat", {"licence_id": "SW-41", "employee_id": "E-333"}]],
     lambda s, i: _final(s, i, changes={
         "seats": _revoked(i, [("SW-41", "E-313"), ("SW-41", "E-325"), ("SW-41", "E-335")]),
         "licences": {"SW-41": {"seats": 9}}},
         new={"seats": [_newseat("SW-41", e) for e in ("E-314", "E-324", "E-331", "E-333")]}),
     ref_errors=[10], mentions=["Zendesk Agent", "Support"])

_TP3_ALLOC = {"LOT-2101": 18000, "LOT-0202": 15000, "LOT-0301": 3000, "LOT-0401": 5000, "LOT-0403": 8500}

_add("TP3", "kitchen", _kitchen(),
     "We need Thai Green Curry with Tofu for the vegan boxes on Tuesday 24 November: plan as many portions as "
     "possible, up to 200, on whichever production line has the most room that day, limited by that room and by the "
     "usable stock of every ingredient. Allocate all of its ingredients following the stock rules and release it to "
     "production. Don't change any other batch.",
     [["get_kitchen_info", {}], ["search_recipes", {"query": "Tofu"}], ["get_recipe", {"recipe_id": "RC-02"}],
      ["list_batches", {"date": "2026-11-24"}], ["list_lots", {"ingredient_id": "ING-21"}],
      ["list_lots", {"ingredient_id": "ING-02"}], ["list_lots", {"ingredient_id": "ING-04"}],
      ["create_batch", {"recipe_id": "RC-02", "portions": 150, "date": "2026-11-24", "line": "L2"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-2101", "grams": 18000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0202", "grams": 15000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0301", "grams": 3000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0401", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0403", "grams": 8500}],
      ["release_batch", {"batch_id": "PB-900"}]],
     [["create_batch", {"recipe_id": "RC-02", "portions": 100, "date": "2026-11-24", "line": "L1"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-2101", "grams": 12000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0202", "grams": 10000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0301", "grams": 2000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0401", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0403", "grams": 4000}],
      ["release_batch", {"batch_id": "PB-900"}]],
     lambda s, i: _final(s, i, changes={"lots": _qty_after(i, _TP3_ALLOC)}, new={"batches": [
         {"recipe": "RC-02", "portions": 150, "date": "2026-11-24", "line": "L2", "status": "released",
          "alloc": _TP3_ALLOC}]}),
     mentions=["Thai Green Curry with Tofu"], seq=[[7, 8, 9, 10, 11, 12, 13]])

_add("TP4", "moderation", _mod(),
     "Please sweep the open reports in c/parenting. Every open self_harm report must be escalated to the tier 2 or "
     "tier 3 moderator who works in the community's language and has the fewest open reports (ties: lower id), and "
     "its post hidden while that happens. Every post with open spam reports from two or more different people is "
     "removed, those spam reports are resolved as actioned, and the post's author gets a strike for it (apply the "
     "suspension threshold if it is reached, signed off by Chen, M-03). Everything else in c/parenting stays as it "
     "is.",
     [["get_mod_policy", {}], ["list_reports", {"community": "c/parenting", "status": "open"}],
      ["list_moderators", {}], ["escalate_report", {"report_id": "RP-3044", "moderator_id": "M-08"}],
      ["set_post_status", {"post_id": "PO-2054", "status": "hidden"}],
      ["set_post_status", {"post_id": "PO-2064", "status": "removed"}],
      ["resolve_report", {"report_id": "RP-3068", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3069", "outcome": "actioned"}],
      ["get_user", {"user_id": "U-165"}],
      ["add_strike", {"user_id": "U-165", "post_id": "PO-2064"}]],
     [["escalate_report", {"report_id": "RP-3044", "moderator_id": "M-03"}],
      ["set_post_status", {"post_id": "PO-2054", "status": "hidden"}],
      ["set_post_status", {"post_id": "PO-2064", "status": "removed"}],
      ["resolve_report", {"report_id": "RP-3068", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3069", "outcome": "actioned"}],
      ["add_strike", {"user_id": "U-165", "post_id": "PO-2064"}],
      ["set_post_status", {"post_id": "PO-2024", "status": "removed"}]],
     lambda s, i: _final(s, i, changes={
         "reports": {"RP-3044": {"status": "escalated", "assignee": "M-08"}, "RP-3068": {"status": "actioned"},
                     "RP-3069": {"status": "actioned"}},
         "posts": {"PO-2054": {"status": "hidden"}, "PO-2064": {"status": "removed"}},
         "users": {"U-165": {"strike_posts": ["PO-2064"]}}}),
     mentions=["c/parenting", "Chen"], seq=[[5, 9]])

_add("TP5", "exams", _exams(),
     "Thirty more candidates have entered for Geography Paper 1, so it will have 68. Update its candidate number - "
     "if its room can't seat 68, first move it to the smallest room that can and is free at that time - and then "
     "add however many support invigilators policy now requires, taking free non-teaching staff in order of staff "
     "id. Nothing else should change.",
     [["get_exam_policy", {}], ["list_exams", {"date": "2027-01-11"}], ["list_rooms", {}],
      ["set_exam_candidates", {"exam_id": "EX-104", "candidates": 68}],
      ["set_exam_room", {"exam_id": "EX-104", "room_id": "RM-02"}],
      ["set_exam_candidates", {"exam_id": "EX-104", "candidates": 68}],
      ["list_staff_schedule", {"date": "2027-01-11"}],
      ["assign_invigilator", {"exam_id": "EX-104", "staff_id": "ST-01", "role": "support"}]],
     [["set_exam_room", {"exam_id": "EX-104", "room_id": "RM-01"}],
      ["set_exam_room", {"exam_id": "EX-104", "room_id": "RM-04"}],
      ["set_exam_candidates", {"exam_id": "EX-104", "candidates": 60}],
      ["assign_invigilator", {"exam_id": "EX-104", "staff_id": "ST-01", "role": "support"}]],
     lambda s, i: _final(s, i, changes={"exams": {"EX-104": {"room": "RM-02", "candidates": 68}}},
                         new={"assignments": [_asg("EX-104", "ST-01", "support")]}),
     ref_errors=[3], mentions=["Geography Paper 1"], seq=[[4, 5, 7]])

_add("TP6", "assets", _assets(),
     "Monitor refresh: every 24-inch monitor still in use that was bought more than five years before today gets "
     "replaced, except for people who are leaving the company. Its user gets one of our in-stock 27-inch monitors "
     "and the old monitor is retired. Hand out the newest 27-inch monitors first, starting with whoever has the "
     "oldest monitor. Nobody should end up with more monitors than before, and nothing else should change.",
     [["get_it_policy", {}], ["list_devices", {"kind": "monitor", "status": "in_use"}],
      ["list_devices", {"kind": "monitor", "status": "in_use", "page": 2}],
      ["list_devices", {"kind": "monitor", "status": "in_stock"}],
      ["find_employees", {"status": "leaving"}],
      ["unassign_device", {"device_id": "D-5003", "destination": "in_stock"}],
      ["retire_device", {"device_id": "D-5003"}],
      ["assign_device", {"device_id": "D-5032", "employee_id": "E-305"}],
      ["unassign_device", {"device_id": "D-5001", "destination": "in_stock"}],
      ["retire_device", {"device_id": "D-5001"}],
      ["assign_device", {"device_id": "D-5035", "employee_id": "E-301"}]],
     [["unassign_device", {"device_id": "D-5003", "destination": "in_stock"}],
      ["retire_device", {"device_id": "D-5003"}],
      ["assign_device", {"device_id": "D-5032", "employee_id": "E-305"}],
      ["unassign_device", {"device_id": "D-5001", "destination": "in_stock"}],
      ["retire_device", {"device_id": "D-5001"}],
      ["assign_device", {"device_id": "D-5035", "employee_id": "E-301"}],
      ["unassign_device", {"device_id": "D-5021", "destination": "in_stock"}],
      ["retire_device", {"device_id": "D-5021"}],
      ["assign_device", {"device_id": "D-5030", "employee_id": "E-312"}]],
     lambda s, i: _final(s, i, changes={"devices": {
         "D-5003": {"status": "retired", "assigned_to": "", "last_user": "E-305"},
         "D-5001": {"status": "retired", "assigned_to": "", "last_user": "E-301"},
         "D-5032": {"status": "in_use", "assigned_to": "E-305"},
         "D-5035": {"status": "in_use", "assigned_to": "E-301"}}}))


# Which difficulty ingredients each task combines (documentation; _verify insists on at least three):
#   a  multi-way conditional decided by data that must be read and aggregated
#   b  business-rule error on the natural first attempt that needs a specific recovery
#   c  ordering dependency (the env rejects, or the outcome changes, in the wrong order)
#   d  batch over several entities picked by a compound criterion, with look-alikes that must stay untouched
#   e  computed argument that depends on two or more earlier read results
#   f  rule that is only stated by a policy read-tool / tool result, not by the prompt
#   g  explicit "nothing else may change" with a tempting side effect
_FEATURES = {
    "T01": "bcdfg", "T02": "bcdfg", "T03": "abceg", "T04": "bcdfg", "T05": "acdg", "T06": "adeg", "T07": "abdfg",
    "T08": "acdfg", "T09": "bcdfg", "T10": "acdefg", "T11": "bcdef", "T12": "abfg", "T13": "bcfg", "T14": "abcdef",
    "T15": "adeg", "T16": "adfg", "T17": "bcef", "T18": "abcdeg", "T19": "adfg", "T20": "abefg", "T21": "bcdfg",
    "T22": "bcdg", "T23": "abcef", "T24": "acefg", "T25": "acdfg", "T26": "acfg", "T27": "adefg", "T28": "adf",
    "T29": "bdfg", "T30": "abfg", "T31": "adg", "T32": "acdfg", "T33": "adef", "T34": "bcef", "T35": "abcef",
    "T36": "abdfg", "T37": "bcef", "T38": "bcef", "T39": "afg", "T40": "bcdg",
    "T41": "bcdfg", "T42": "acdf", "T43": "adg", "T44": "bcfg", "T45": "abcfg", "T46": "acdf", "T47": "abcfg",
    "T48": "acdefg", "T49": "adfg", "T50": "acdeg", "T51": "adfg", "T52": "abcdef", "T53": "bcfg", "T54": "adeg",
    "T55": "acefg", "T56": "abcef", "T57": "acefg", "T58": "abdeg", "T59": "abcdef", "T60": "acdef", "T61": "bcef",
    "T62": "adefg", "T63": "adeg", "T64": "adef", "T65": "adg", "T66": "adg", "T67": "adfg", "T68": "acdef",
    "T69": "adef", "T70": "bcef", "T71": "adef", "T72": "acdg", "T73": "abef", "T74": "adef", "T75": "bcg",
    "TP1": "bcdg", "TP2": "acdef", "TP3": "aef", "TP4": "adf", "TP5": "bcef", "TP6": "acde",
}
for _t in TASKS:
    _t["features"] = _FEATURES[_t["id"]]


# --------------------------------------------------------------------------
# self-verification
# --------------------------------------------------------------------------
def _is_err(res):
    return isinstance(res, dict) and "error" in res


def _strings(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.add(k)
            _strings(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _strings(v, acc)
    elif isinstance(obj, str):
        acc.add(obj)


_GARBAGE = [None, "", "zzz", "/", "HD-501; drop", -1, 0, 3, 10 ** 12, 1.5, 2.0, float("inf"), float("nan"), True, False,
            [], [None], [{}], ["a", 1], [[1]], {}, {"x": {"y": [1]}}, {"id": 1, "qty": "2"},
            "2026-13-45", "25:99", "2026-11-04", "2026-11-25", "2027-01-15", "09:00", "9:00", "é中", "null"]


def _safe_call(env, name, args, label):
    before = _clone(env.state)
    try:
        res = env.call(name, args)
    except Exception as e:  # pragma: no cover - this is what the fuzz is for
        raise AssertionError("%s: call raised %r for %r %r" % (label, e, name, args))
    json.dumps(res)
    if _is_err(res):
        assert env.state == before, "%s: state changed on error for %r %r" % (label, name, args)
    if _is_read(name):
        assert env.state == before, "%s: read tool %r changed the state" % (label, name)
    json.dumps(env.state)
    return res


def _fuzz():
    rng = random.Random(20261104)
    calls = 0
    for ename in sorted(ENVS):
        cls = ENVS[ename]
        inits = [t["init"] for t in TASKS if t["env"] == ename]
        pool = set()
        for ini in inits:
            _strings(ini, pool)
        pool = sorted(pool)
        dates = [x for x in pool if re.match(r"^\d{4}-\d{2}-\d{2}$", x)] or ["2026-11-04"]
        names = [t["function"]["name"] for t in cls.TOOLS]
        env = cls(inits[0])
        for name in names + [None, 5, "nope", "", [], {}, 1.5, "fill_hold; drop"]:
            for args in _GARBAGE + ["{}", 7]:
                _safe_call(env, name, args, ename)
                calls += 1

        def rand_value(schema, depth=0):
            r = rng.random()
            if r < 0.25 or depth > 3:
                return rng.choice(_GARBAGE)
            t = schema.get("type")
            if t == "string":
                if "enum" in schema and rng.random() < 0.85:
                    return rng.choice(schema["enum"])
                return rng.choice(pool)
            if t == "integer":
                return rng.choice([-5, 0, 1, 2, 3, 6, 8, 9, 10, 12, 15, 20, 46, 100, 150, 1000, 5000, 19200])
            if t == "boolean":
                return rng.random() < 0.5
            if t == "array":
                return [rand_value(schema["items"], depth + 1) for _ in range(rng.randint(0, 3))]
            if t == "object":
                out = {}
                for k, sub in schema.get("properties", {}).items():
                    if k in schema.get("required", []) or rng.random() < 0.5:
                        out[k] = rand_value(sub, depth + 1)
                if rng.random() < 0.05:
                    out["bogus"] = 1
                return out
            return None

        def typed_value(key, schema):
            """A plausible value for a parameter: ids of the right family, so that many calls succeed."""
            if "enum" in schema:
                return rng.choice(schema["enum"])
            t = schema.get("type")
            if t == "boolean":
                return rng.random() < 0.5
            if t == "integer":
                return rng.choice([1, 2, 3, 5, 8, 9, 10, 12, 15, 20, 50, 200, 1000, 4000])
            if "date" in key or key == "expires_before":
                return rng.choice(dates)
            if key == "start":
                return rng.choice(["09:00", "13:30", "11:00"])
            return rng.choice(pool)

        ok_writes = 0
        for n in range(3000):
            if n % 150 == 0:
                env = cls(rng.choice(inits))
            tool = rng.choice(cls.TOOLS)["function"]
            if n % 2:
                args = rand_value(tool["parameters"])
            else:
                props = tool["parameters"]["properties"]
                args = dict((k, typed_value(k, sub)) for k, sub in props.items()
                            if k in tool["parameters"]["required"] or rng.random() < 0.3)
            res = _safe_call(env, tool["name"], args, ename)
            if not _is_err(res) and not _is_read(tool["name"]):
                ok_writes += 1
            calls += 1
        assert ok_writes > 0, (ename, "random fuzz never produced a successful write")
    # mutate every argument of every reference call with every garbage value, and delete each argument
    for t in TASKS:
        base = ENVS[t["env"]](t["init"])          # state just before each reference step
        for step, (name, args) in enumerate(t["reference"]):
            for key in list(args) + ["__extra__"]:
                for g in _GARBAGE:
                    env = ENVS[t["env"]](base.state)
                    bad = copy.deepcopy(args)
                    bad[key] = g
                    _safe_call(env, name, bad, t["id"])
                    calls += 1
            for key in args:
                env = ENVS[t["env"]](base.state)
                bad = copy.deepcopy(args)
                del bad[key]
                _safe_call(env, name, bad, t["id"])
                calls += 1
            base.call(name, copy.deepcopy(args))
    return calls


def _schema_ok(schema, path):
    """No union types, no null, only simple types; arrays of scalars and flat objects only."""
    t = schema.get("type")
    assert isinstance(t, str), (path, "type must be a single string, got %r" % (t,))
    assert t in ("string", "integer", "boolean", "array", "object"), (path, t)
    for bad in ("anyOf", "oneOf", "allOf", "nullable"):
        assert bad not in schema, (path, bad)
    if "enum" in schema:
        assert all(isinstance(x, str) and x for x in schema["enum"]), (path, "enum")
    if t == "array":
        assert schema["items"]["type"] in ("string", "integer", "boolean"), (path, "array items must be scalars")
    if t == "object":
        for k, sub in schema.get("properties", {}).items():
            _schema_ok(sub, path + "." + k)


def _no_empty(obj, where):
    if obj is None or obj == "":
        raise AssertionError("%s: reference passes None or an empty string" % (where,))
    if isinstance(obj, dict):
        for v in obj.values():
            _no_empty(v, where)
    elif isinstance(obj, list):
        for v in obj:
            _no_empty(v, where)


_ID_PATTERNS = [r"\b[A-Z]{1,5}\d{0,3}-[A-Z0-9]{1,5}\b",     # P-201, HD-501, CP-1101, LOT-0102, RM-01, ST-07
                r"\b[A-Z]{2,5}\d{3}\b",
                r"\b[A-Z]{4,}\b"]                           # anything written in capitals


def _prompt_entities(t):
    """Every id-like token, quoted name and declared mention in the prompt must exist in init."""
    pool = set()
    _strings(t["init"], pool)
    blob = "\n".join(sorted(pool))
    vocab = set()                      # enum values of the env's tool schemas may be quoted too
    _strings(ENVS[t["env"]].TOOLS, vocab)
    prompt = t["prompt"]
    found = set()
    for pat in _ID_PATTERNS:
        for tok in re.findall(pat, prompt):
            found.add(tok)
            assert tok in pool or tok in t["creates"], (t["id"], "prompt names %r which is not in init" % tok)
    for tok in re.findall(r"(?<![A-Za-z0-9])'([^']{2,40})'(?![A-Za-z0-9])", prompt):
        found.add(tok)
        assert tok in pool or tok in vocab or tok in t["creates"], (t["id"], "prompt quotes %r which is not in init" % tok)
    for tok in re.findall(r"@([A-Za-z0-9_]+)", prompt):
        found.add(tok)
        assert tok in pool, (t["id"], "prompt names user @%s who is not in init" % tok)
    for tok in re.findall(r"\bc/[a-z-]+", prompt):
        found.add(tok)
        assert tok in pool, (t["id"], "prompt names community %r which is not in init" % tok)
    for m in t["mentions"]:
        assert m in prompt, (t["id"], "declared mention %r is not in the prompt" % m)
        assert m in blob, (t["id"], "prompt mentions %r which is not in init" % m)
    return len(found) + len(t["mentions"])


def _alt_orders(t, rng, tries=300):
    """Replay the reference's successful WRITE calls in other orders. An order is 'valid' if no call errors;
    every valid order must satisfy check. Orders that break a declared seq group are not tried."""
    writes = [n for n, (name, _) in enumerate(t["reference"]) if not _is_read(name) and n not in t["ref_errors"]]
    orders, seen = [list(reversed(writes))], set()
    for _ in range(tries):
        p = list(writes)
        rng.shuffle(p)
        orders.append(p)
    for n in range(len(writes) - 1):       # every adjacent swap of the reference order
        p = list(writes)
        p[n], p[n + 1] = p[n + 1], p[n]
        orders.append(p)
    valid = 0
    for p in orders:
        if tuple(p) in seen or p == writes:
            continue
        seen.add(tuple(p))
        if any([x for x in p if x in g] != [x for x in writes if x in g] for g in t["seq"]):
            continue
        env = ENVS[t["env"]](t["init"])
        ok = True
        for n in p:
            name, args = t["reference"][n]
            if _is_err(env.call(name, copy.deepcopy(args))):
                ok = False
                break
        if not ok:
            continue
        valid += 1
        assert t["check"](env.state) is True, (t["id"], "valid alternative write order fails check", p)
    return valid, len(writes)


# Hand-written alternative ROUTES (different calls, not just a different order) that are equally valid
# solutions; each must satisfy check. Guards against checks that only fit the reference's route.
_ALT_ROUTES = {
    "T01": [["fill_hold", {"hold_id": "HD-501", "copy_id": "CP-1107"}],
            ["fill_hold", {"hold_id": "HD-504", "copy_id": "CP-1102"}],
            ["fill_hold", {"hold_id": "HD-504", "copy_id": "CP-1101"}],
            ["fill_hold", {"hold_id": "HD-506", "copy_id": "CP-1103"}]],
    "T04": [["set_copy_status", {"copy_id": "CP-1203", "status": "withdrawn"}],
            ["receive_copy", {"copy_id": "CP-1203"}], ["set_copy_status", {"copy_id": "CP-1203", "status": "withdrawn"}],
            ["receive_copy", {"copy_id": "CP-1701"}], ["fill_hold", {"hold_id": "HD-542", "copy_id": "CP-1701"}],
            ["receive_copy", {"copy_id": "CP-1611"}], ["receive_copy", {"copy_id": "CP-1301"}]],
    "T09": [["unassign_device", {"device_id": "D-4011", "destination": "repair"}], ["retire_device", {"device_id": "D-4011"}],
            ["revoke_seat", {"licence_id": "SW-52", "employee_id": "E-312"}],
            ["unassign_device", {"device_id": "D-6001", "destination": "repair"}],
            ["revoke_seat", {"licence_id": "SW-23", "employee_id": "E-312"}],
            ["unassign_device", {"device_id": "D-5021", "destination": "in_stock"}],
            ["unassign_device", {"device_id": "D-5020", "destination": "in_stock"}],
            ["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-312"}],
            ["update_employee", {"employee_id": "E-312", "status": "left"}]],
    "T11": [["assign_device", {"device_id": "D-4032", "employee_id": "E-340"}],
            ["assign_device", {"device_id": "D-4030", "employee_id": "E-341"}],
            ["set_seat_count", {"licence_id": "SW-23", "seats": 7}], ["set_seat_count", {"licence_id": "SW-23", "seats": 8}],
            ["set_seat_count", {"licence_id": "SW-52", "seats": 11}],
            ["assign_seat", {"licence_id": "SW-52", "employee_id": "E-341"}],
            ["assign_seat", {"licence_id": "SW-52", "employee_id": "E-340"}],
            ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-341"}],
            ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-340"}],
            ["assign_seat", {"licence_id": "SW-11", "employee_id": "E-341"}],
            ["assign_seat", {"licence_id": "SW-11", "employee_id": "E-340"}]],
    "T14": [["unassign_device", {"device_id": "D-4020", "destination": "repair"}],
            ["assign_device", {"device_id": "D-4030", "employee_id": "E-322"}],
            ["unassign_device", {"device_id": "D-4016", "destination": "in_stock"}],
            ["assign_device", {"device_id": "D-4033", "employee_id": "E-318"}],
            ["unassign_device", {"device_id": "D-4019", "destination": "in_stock"}],
            ["assign_device", {"device_id": "D-4032", "employee_id": "E-321"}],
            ["retire_device", {"device_id": "D-4016"}], ["retire_device", {"device_id": "D-4019"}],
            ["retire_device", {"device_id": "D-4020"}]],
    "T18": [["deallocate_lot", {"batch_id": "PB-843", "lot_id": "LOT-0602"}],
            ["update_batch", {"batch_id": "PB-843", "portions": 46}],
            ["deallocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0601"}],
            ["quarantine_lot", {"lot_id": "LOT-0602", "reason": "recall"}],
            ["quarantine_lot", {"lot_id": "LOT-0601", "reason": "recall"}],
            ["allocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0604", "grams": 1000}],
            ["allocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0604", "grams": 200}],
            ["allocate_lot", {"batch_id": "PB-833", "lot_id": "LOT-0603", "grams": 800}],
            ["allocate_lot", {"batch_id": "PB-843", "lot_id": "LOT-0603", "grams": 690}]],
    "T23": [["cancel_batch", {"batch_id": "PB-830"}], ["update_batch", {"batch_id": "PB-832", "portions": 200}],
            ["deallocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1302"}],
            ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1302", "grams": 16000}],
            ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1201", "grams": 10800}],
            ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1401", "grams": 1800}],
            ["release_batch", {"batch_id": "PB-832"}]],
    "T23b": [["update_batch", {"batch_id": "PB-832", "portions": 200}],
             ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1202", "grams": 10800}],
             ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1303", "grams": 9600}],
             ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1401", "grams": 1000}],
             ["allocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1402", "grams": 800}],
             ["release_batch", {"batch_id": "PB-832"}], ["cancel_batch", {"batch_id": "PB-830"}]],
    "T27": [["set_post_status", {"post_id": "PO-2021", "status": "removed"}],
            ["add_strike", {"user_id": "U-120", "post_id": "PO-2021"}],
            ["set_post_status", {"post_id": "PO-2020", "status": "removed"}],
            ["add_strike", {"user_id": "U-120", "post_id": "PO-2020"}],
            ["remove_strike", {"user_id": "U-120", "post_id": "PO-2020"}],
            ["resolve_report", {"report_id": "RP-3021", "outcome": "actioned"}],
            ["resolve_report", {"report_id": "RP-3020", "outcome": "actioned"}],
            ["set_post_status", {"post_id": "PO-2024", "status": "removed"}],
            ["add_strike", {"user_id": "U-123", "post_id": "PO-2024"}],
            ["resolve_report", {"report_id": "RP-3024", "outcome": "actioned"}],
            ["assign_report", {"report_id": "RP-3023", "moderator_id": "M-02"}],
            ["assign_report", {"report_id": "RP-3022", "moderator_id": "M-02"}],
            ["assign_report", {"report_id": "RP-3022", "moderator_id": "M-01"}]],
    "T35": [["split_exam", {"exam_id": "EX-133", "room_id": "RM-05", "candidates": 15}],
            ["assign_invigilator", {"exam_id": "EX-900", "staff_id": "ST-03", "role": "support"}],
            ["assign_invigilator", {"exam_id": "EX-900", "staff_id": "ST-02", "role": "lead"}],
            ["set_exam_room", {"exam_id": "EX-133", "room_id": "RM-04"}],
            ["set_exam_room", {"exam_id": "EX-131", "room_id": "RM-01"}]],
    "T42": [["fill_hold", {"hold_id": "HD-501", "copy_id": "CP-1107"}],
            ["fill_hold", {"hold_id": "HD-504", "copy_id": "CP-1102"}],
            ["fill_hold", {"hold_id": "HD-504", "copy_id": "CP-1101"}],
            ["fill_hold", {"hold_id": "HD-551", "copy_id": "CP-1503"}],
            ["fill_hold", {"hold_id": "HD-533", "copy_id": "CP-1403"}]],
    "T53": [["unassign_device", {"device_id": "D-4015", "destination": "in_stock"}],
            ["assign_device", {"device_id": "D-4032", "employee_id": "E-317"}],
            ["unassign_device", {"device_id": "D-4014", "destination": "in_stock"}],
            ["assign_device", {"device_id": "D-4015", "employee_id": "E-315"}],
            ["unassign_device", {"device_id": "D-4010", "destination": "in_stock"}],
            ["assign_device", {"device_id": "D-4014", "employee_id": "E-311"}],
            ["retire_device", {"device_id": "D-4010"}]],
    "T75": [["set_exam_room", {"exam_id": "EX-141", "room_id": "RM-02"}],
            ["set_exam_room", {"exam_id": "EX-140", "room_id": "RM-04"}],
            ["set_exam_room", {"exam_id": "EX-141", "room_id": "RM-01"}],
            ["set_exam_room", {"exam_id": "EX-144", "room_id": "RM-06"}],
            ["set_exam_room", {"exam_id": "EX-144", "room_id": "RM-02"}],
            ["set_exam_room", {"exam_id": "EX-143", "room_id": "RM-04"}],
            ["set_exam_room", {"exam_id": "EX-144", "room_id": "RM-03"}]],
    "T40": [["set_invigilator_role", {"exam_id": "EX-134", "staff_id": "ST-11", "role": "support"}],
            ["set_invigilator_role", {"exam_id": "EX-123", "staff_id": "ST-09", "role": "support"}],
            ["set_invigilator_role", {"exam_id": "EX-101", "staff_id": "ST-14", "role": "support"}],
            ["set_invigilator_role", {"exam_id": "EX-134", "staff_id": "ST-01", "role": "lead"}],
            ["set_invigilator_role", {"exam_id": "EX-123", "staff_id": "ST-01", "role": "lead"}],
            ["set_invigilator_role", {"exam_id": "EX-101", "staff_id": "ST-16", "role": "lead"}]],
}


def _verify(expect_ids=None):
    ids = [t["id"] for t in TASKS]
    if expect_ids is not None:
        assert ids == expect_ids, ids
    assert len(set(ids)) == len(ids)
    by_id = dict((t["id"], t) for t in TASKS)
    for ename, cls in ENVS.items():
        assert 9 <= len(cls.TOOLS) <= 14, (ename, len(cls.TOOLS))
        json.dumps(cls.TOOLS)
        names = [tool["function"]["name"] for tool in cls.TOOLS]
        assert len(set(names)) == len(names)
        assert any(n.startswith("get_") and n.endswith(("_info", "_policy")) for n in names), (ename, "policy tool")
        assert any("page" in tool["function"]["parameters"]["properties"] for tool in cls.TOOLS), (ename, "paging")
        for tool in cls.TOOLS:
            fn = tool["function"]
            assert tool["type"] == "function" and fn["parameters"]["type"] == "object"
            assert hasattr(cls, "_t_" + fn["name"]), fn["name"]
            _schema_ok(fn["parameters"], fn["name"])
            for r in fn["parameters"]["required"]:
                assert r in fn["parameters"]["properties"], (fn["name"], r)
    ok = 0
    rng = random.Random(7)
    report = []
    for t in TASKS:
        tid = t["id"]
        cls = ENVS[t["env"]]
        tools = dict((x["function"]["name"], x["function"]) for x in cls.TOOLS)
        words = len(t["prompt"].split())
        assert 60 <= words <= 200, (tid, "prompt words", words)
        assert 7 <= len(t["reference"]) <= 16, (tid, "reference length", len(t["reference"]))
        assert len(set(t["features"])) >= 3 and set(t["features"]) <= set("abcdefg"), (tid, "features")
        json.dumps(t["init"])
        json.dumps(t["reference"])
        json.dumps(t["trap"])
        _no_empty(t["reference"], tid)
        n_ent = _prompt_entities(t)
        # (a) untouched initial state must fail
        env = cls(t["init"])
        assert env.state == t["init"] and env.state is not t["init"]
        assert t["check"](env.state) is False, (tid, "check passes on the initial state")
        # (b) reference solves it; errors only where declared; reads never change the state
        chars = 0
        for n, (name, args) in enumerate(t["reference"]):
            assert name in tools, (tid, "unknown tool in reference", name)
            before = copy.deepcopy(env.state)
            res = env.call(name, copy.deepcopy(args))
            chars += len(json.dumps(res))
            assert _is_err(res) == (n in t["ref_errors"]), (tid, "reference step", n, name, res)
            if _is_read(name) or _is_err(res):
                assert env.state == before, (tid, "state changed by read/error step", n)
            else:
                assert env.state != before, (tid, "reference write changed nothing", n, name)
        assert t["check"](env.state) is True, (tid, "reference does not satisfy check")
        assert chars < 45000, (tid, "reference transcript too long for the context window", chars)
        json.dumps(env.state)
        # every read tool that takes no required arguments must leave a solved state solved
        for fn in tools.values():
            if not fn["parameters"]["required"] and _is_read(fn["name"]):
                env.call(fn["name"], {})
        assert t["check"](env.state) is True, (tid, "read tools changed the state")
        # (d) trap must change something and fail
        env = cls(t["init"])
        for name, args in t["trap"]:
            assert name in tools, (tid, "unknown tool in trap", name)
            json.dumps(env.call(name, copy.deepcopy(args)))
        assert env.state != t["init"], (tid, "trap changed nothing")
        assert t["check"](env.state) is False, (tid, "trap passes the check")
        # (e) dropping any single successful write from the reference must fail
        writes = [n for n, (name, _) in enumerate(t["reference"]) if not _is_read(name) and n not in t["ref_errors"]]
        for skip in writes:
            env = cls(t["init"])
            for n, (name, args) in enumerate(t["reference"]):
                if n != skip:
                    env.call(name, copy.deepcopy(args))
            assert t["check"](env.state) is False, (tid, "check passes although reference step %d was skipped" % skip)
        # (f) alternative valid write orders must pass
        valid, nw = _alt_orders(t, rng)
        report.append((tid, len(t["reference"]), nw, valid, words, n_ent, chars))
        # init must not have been mutated by any of the above
        assert t["check"](copy.deepcopy(t["init"])) is False
        ok += 1
    # (g) hand-written alternative routes must pass as well
    for key, route in sorted(_ALT_ROUTES.items()):
        t = by_id[key.rstrip("b")]
        env = ENVS[t["env"]](t["init"])
        for name, args in route:
            env.call(name, copy.deepcopy(args))   # may include a refused attempt, as a model's route would
        assert t["check"](env.state) is True, (key, "alternative route fails check")
    calls = _fuzz()
    for t in TASKS:   # fuzz must not have leaked into the fixtures
        env = ENVS[t["env"]](t["init"])
        for name, args in t["reference"]:
            env.call(name, copy.deepcopy(args))
        assert t["check"](env.state) is True, (t["id"], "fixture mutated by fuzz")
    return ok, calls, report


if __name__ == "__main__":
    _expect = ["T%02d" % n for n in range(1, 76)] + ["TP%d" % n for n in range(1, 7)]
    _ok, _calls, _report = _verify(_expect)
    for _e in sorted(ENVS):
        assert len([t for t in TASKS[:40] if t["env"] == _e]) == 8, _e
        assert len([t for t in TASKS[40:75] if t["env"] == _e]) == 7, _e
        assert len([t for t in TASKS[75:] if t["env"] == _e]) >= 1, _e
    for _t in TASKS:   # sizes of the initial states
        _n = sum(len(v) for v in _t["init"].values() if _is_collection(v))
        assert 20 <= _n, (_t["id"], _n)
    print("envs: " + ", ".join("%s (%d tools)" % (k, len(v.TOOLS)) for k, v in sorted(ENVS.items())))
    print("reference calls per task: min %d, max %d; prompt words: min %d, max %d" % (
        min(r[1] for r in _report), max(r[1] for r in _report), min(r[4] for r in _report), max(r[4] for r in _report)))
    print("alternative write orders: %d valid orders replayed over %d tasks, all pass check; tasks with none: %s" % (
        sum(r[3] for r in _report), len(_report), ", ".join(r[0] for r in _report if r[3] == 0) or "-"))
    print("alternative routes (different calls, same outcome): %d routes, all pass check" % len(_ALT_ROUTES))
    print("fuzz: %d calls, 0 exceptions, state unchanged on every error" % _calls)
    print("OK %d/%d" % (_ok, len(TASKS)))
