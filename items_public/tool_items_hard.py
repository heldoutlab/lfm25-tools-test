# -*- coding: utf-8 -*-
"""HARD supplement to the lfm25-tools public task set (pre-registration amendment).

Stdlib only; runs on Python 3.9 and 3.12.  Reuses the five mock environments of
tool_items.py (same folder, imported unchanged; that file is frozen) and builds
per-task variant fixtures from its base fixtures (deep copies).  Tasks are graded
on the FINAL STATE of the environment (task["check"](env.state)).

Layout
  ENVS   = tool_items.ENVS (library, assets, kitchen, moderation, exams; 14 tools each)
  TASKS  H01..H25 (test set, 5 per environment) then HP1..HP3 (calibration, same
         difficulty, three different environments)

Every task needs 12-20 reference calls and combines at least four difficulty
ingredients, always including P, R, S and one of B/C:
  P  a rule found only via the policy tool (or a tool error) that changes the natural plan
  R  a mid-task rule error on the natural attempt that forces re-planning
  S  state that must be re-read after an earlier write changed it
  B  a batch over entities picked by a compound criterion, with look-alikes that must stay untouched
  C  a computed argument that depends on two or more earlier reads
  N  the correct end state requires deliberately NOT doing something the prompt makes tempting
  O  an ordering dependency

Run this file directly to self-verify:  prints "OK 28/28".
"""
import copy
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tool_items as base  # noqa: E402  (frozen; never modified)

ENVS = base.ENVS
_final = base._final
_any = base._any
_RDY = base._RDY
_CANC = base._CANC
_is_read = base._is_read
_is_err = base._is_err

TASKS = []


def _add(tid, env, init, prompt, reference, trap, check, features, ref_errors=(), mentions=(), creates=(), seq=()):
    frozen = copy.deepcopy(init)
    TASKS.append({
        "id": tid, "env": env, "init": init, "prompt": prompt,
        "reference": reference, "ref_errors": list(ref_errors), "trap": trap, "features": features,
        "mentions": list(mentions), "creates": list(creates), "seq": [list(g) for g in seq],
        "check": (lambda state, _c=check, _i=frozen: bool(_c(state, _i))),
    })


def _new_ids(s, i, coll):
    return sorted(k for k in s.get(coll, {}) if k not in i[coll])


# --------------------------------------------------------------------------
# library H01-H05
# --------------------------------------------------------------------------
_hold = base._hold_rec
_copy = base._copy_rec


def _one_new(s, i, coll, pred):
    ids = [k for k in _new_ids(s, i, coll) if pred(s[coll][k])]
    return ids[0] if len(ids) == 1 else None


def _h01(s, i):
    try:
        lp = _one_new(s, i, "holds", lambda h: h.get("title") == "BK-111")
    except (AttributeError, TypeError):
        return False
    if lp is None:
        return False
    return _final(s, i, changes={
        "holds": {"HD-582": _CANC, "HD-505": {"pickup": "NOR"}},
        "copies": {"CP-1111": {"status": "on_hold_shelf", "hold": lp}}},
        new={"holds": [
            {"patron": "P-206", "title": "BK-110", "pickup": "NOR", "status": "waiting", "placed": "2026-11-04",
             "copy": "", "ready_until": ""},
            {"patron": "P-209", "title": "BK-111", "pickup": "NOR", "status": "ready", "placed": "2026-11-04",
             "copy": "CP-1111", "ready_until": _RDY},
            {"patron": "P-214", "title": "BK-110", "pickup": "NOR", "status": "waiting", "placed": "2026-11-04",
             "copy": "", "ready_until": ""}]})


_add("H01", "library", base._lib(),
     "The Northgate reading group is doing 'The Salt Road' next month. The members are Mia Okonkwo-Reed, Yusuf "
     "Demir, Jonah Reyes, Lena Fischbach and Amara Osei. Each member should end up with a hold for collection at "
     "Northgate on the regular book; if the rules won't allow that for someone, give them a hold on the large-print "
     "edition instead, if the rules allow that. A member who already has a hold on the regular book for another "
     "branch has it switched to Northgate rather than getting a second one. If a member's only obstacle is their "
     "hold limit, cancel their newest waiting hold (not a suspended or ready one) to make room - never cancel "
     "anything otherwise. Finally, if a large-print copy on the open shelf at Northgate may go to one of the new "
     "large-print holds under the queue rules, fill it. Nothing else should change.",
     [["search_titles", {"query": "The Salt Road"}], ["find_patrons", {"query": "Okonkwo"}],
      ["find_patrons", {"query": "Yusuf Demir"}], ["find_patrons", {"query": "Jonah Reyes"}],
      ["find_patrons", {"query": "Lena Fischbach"}], ["find_patrons", {"query": "Amara Osei"}],
      ["list_holds", {"title_id": "BK-110"}], ["list_holds", {"title_id": "BK-110", "page": 2}],
      ["place_hold", {"patron_id": "P-206", "title_id": "BK-110", "pickup_branch": "NOR"}],
      ["place_hold", {"patron_id": "P-209", "title_id": "BK-110", "pickup_branch": "NOR"}],
      ["place_hold", {"patron_id": "P-209", "title_id": "BK-111", "pickup_branch": "NOR"}],
      ["update_hold", {"hold_id": "HD-505", "pickup_branch": "NOR"}],
      ["place_hold", {"patron_id": "P-214", "title_id": "BK-110", "pickup_branch": "NOR"}],
      ["list_holds", {"patron_id": "P-214"}], ["cancel_hold", {"hold_id": "HD-582"}],
      ["place_hold", {"patron_id": "P-214", "title_id": "BK-110", "pickup_branch": "NOR"}],
      ["place_hold", {"patron_id": "P-204", "title_id": "BK-110", "pickup_branch": "NOR"}],
      ["list_holds", {"title_id": "BK-111"}], ["fill_hold", {"hold_id": "HD-601", "copy_id": "CP-1111"}]],
     [["place_hold", {"patron_id": "P-206", "title_id": "BK-110", "pickup_branch": "NOR"}],
      ["update_hold", {"hold_id": "HD-505", "pickup_branch": "NOR"}],
      ["cancel_hold", {"hold_id": "HD-583"}], ["cancel_hold", {"hold_id": "HD-582"}],
      ["place_hold", {"patron_id": "P-214", "title_id": "BK-110", "pickup_branch": "NOR"}],
      ["place_hold", {"patron_id": "P-209", "title_id": "BK-111", "pickup_branch": "NOR"}]],
     _h01, "PRSBO", ref_errors=[9, 12, 16],
     mentions=["Mia Okonkwo-Reed", "Yusuf Demir", "Jonah Reyes", "Lena Fischbach", "Amara Osei", "Northgate"],
     seq=[[8, 10, 15, 18]])


def _lib_h02():
    st = base._lib()
    st["holds"]["HD-584"] = _hold("P-212", "BK-150", "RIV", "suspended", "10-10")
    st["holds"]["HD-585"] = _hold("P-212", "BK-121", "RIV", "ready", "10-20", "CP-1214", "2026-11-07")
    st["copies"]["CP-1214"] = _copy("BK-121", "RIV", "on_hold_shelf", hold="HD-585")
    return st


def _h02(s, i):
    try:
        a = _one_new(s, i, "holds", lambda h: h.get("title") == "BK-130")
        b = _one_new(s, i, "holds", lambda h: h.get("title") == "BK-121")
    except (AttributeError, TypeError):
        return False
    if a is None or b is None:
        return False
    return _final(s, i, changes={
        "holds": {"HD-504": {"pickup": "NOR"}, "HD-584": {"pickup": "NOR"}, "HD-540": _CANC, "HD-585": _CANC},
        "copies": {"CP-1301": {"location": "RIV", "destination": "NOR", "status": "in_transit", "hold": a},
                   "CP-1214": {"destination": "NOR", "status": "in_transit", "hold": b}}},
        new={"holds": [
            {"patron": "P-212", "title": "BK-130", "pickup": "NOR", "status": "in_transit", "placed": "2026-11-04",
             "copy": "CP-1301", "ready_until": ""},
            {"patron": "P-212", "title": "BK-121", "pickup": "NOR", "status": "in_transit", "placed": "2026-11-04",
             "copy": "CP-1214", "ready_until": ""}]})


_add("H02", "library", _lib_h02(),
     "Priya Nair has moved house and from now on collects everything at Northgate. Move each of her active holds to "
     "Northgate. Where the rules won't let a hold simply be redirected, get it there another way: a copy that is "
     "still travelling has to be checked in at its destination first, and a hold that can't be redirected is "
     "cancelled and placed again for Northgate (she accepts going to the back of the queue). Suspended holds must "
     "still be suspended at the end. Once all her holds are sorted, every copy freed along the way that is still "
     "on the open shelf goes to the next hold in line under the queue rules. Nothing else should change.",
     [["find_patrons", {"query": "Priya Nair"}], ["list_holds", {"patron_id": "P-212"}],
      ["update_hold", {"hold_id": "HD-504", "pickup_branch": "NOR"}],
      ["update_hold", {"hold_id": "HD-540", "pickup_branch": "NOR"}],
      ["receive_copy", {"copy_id": "CP-1301"}], ["cancel_hold", {"hold_id": "HD-540"}],
      ["place_hold", {"patron_id": "P-212", "title_id": "BK-130", "pickup_branch": "NOR"}],
      ["set_hold_suspended", {"hold_id": "HD-584", "suspended": False}],
      ["update_hold", {"hold_id": "HD-584", "pickup_branch": "NOR"}],
      ["set_hold_suspended", {"hold_id": "HD-584", "suspended": True}],
      ["cancel_hold", {"hold_id": "HD-585"}],
      ["place_hold", {"patron_id": "P-212", "title_id": "BK-121", "pickup_branch": "NOR"}],
      ["list_holds", {"title_id": "BK-130"}], ["fill_hold", {"hold_id": "HD-600", "copy_id": "CP-1301"}],
      ["list_holds", {"title_id": "BK-121"}], ["fill_hold", {"hold_id": "HD-601", "copy_id": "CP-1214"}]],
     [["update_hold", {"hold_id": "HD-504", "pickup_branch": "NOR"}],
      ["receive_copy", {"copy_id": "CP-1301"}], ["cancel_hold", {"hold_id": "HD-540"}],
      ["place_hold", {"patron_id": "P-212", "title_id": "BK-130", "pickup_branch": "NOR"}],
      ["set_hold_suspended", {"hold_id": "HD-584", "suspended": False}],
      ["update_hold", {"hold_id": "HD-584", "pickup_branch": "NOR"}],
      ["set_hold_suspended", {"hold_id": "HD-584", "suspended": True}],
      ["cancel_hold", {"hold_id": "HD-585"}],
      ["place_hold", {"patron_id": "P-212", "title_id": "BK-121", "pickup_branch": "NOR"}]],
     _h02, "PRSCO", ref_errors=[3], mentions=["Priya Nair", "Northgate"],
     seq=[[4, 5, 6, 11, 13, 15], [10, 11]])


def _lib_h03():
    st = base._lib()
    st["copies"]["CP-1114"] = _copy("BK-111", "CEN", "on_shelf")
    st["copies"]["CP-1115"] = _copy("BK-111", "CEN", "on_loan", borrower="P-211", due="2026-11-18")
    st["holds"]["HD-586"] = _hold("P-201", "BK-110", "CEN", "waiting", "11-01")
    return st


def _h03(s, i):
    try:
        a = _one_new(s, i, "holds", lambda h: h.get("patron") == "P-217")
        b = _one_new(s, i, "holds", lambda h: h.get("patron") == "P-201")
    except (AttributeError, TypeError):
        return False
    if a is None or b is None:
        return False
    return _final(s, i, changes={
        "holds": {"HD-550": _CANC, "HD-586": _CANC},
        "copies": {"CP-1111": {"status": "on_hold_shelf", "hold": a}, "CP-1114": {"status": "on_hold_shelf", "hold": b}}},
        new={"holds": [
            {"patron": "P-217", "title": "BK-111", "pickup": "NOR", "status": "ready", "placed": "2026-11-04",
             "copy": "CP-1111", "ready_until": _RDY},
            {"patron": "P-201", "title": "BK-111", "pickup": "CEN", "status": "ready", "placed": "2026-11-04",
             "copy": "CP-1114", "ready_until": _RDY}]})


_add("H03", "library", _lib_h03(),
     "The waiting list for 'The Salt Road' is long while large-print copies sit on our shelves. Offer the large "
     "print to everyone who is fourth or further back in the queue for the regular book - counting only the holds "
     "that the queue rules would actually serve, in queue order. Switch each of them, in queue order: they get a "
     "hold on the large-print edition with the same pickup branch and their regular-book hold is cancelled. Anyone "
     "who can't be given the large-print hold keeps their regular-book hold exactly as it is. Then fill the new "
     "large-print holds, following the queue rules, from large-print copies on the open shelf at each hold's own "
     "pickup branch - no copy travels. Nothing else should change.",
     [["get_library_info", {}], ["search_titles", {"query": "The Salt Road"}],
      ["list_holds", {"title_id": "BK-110"}], ["list_holds", {"title_id": "BK-110", "page": 2}],
      ["get_patron", {"patron_id": "P-204"}],
      ["place_hold", {"patron_id": "P-211", "title_id": "BK-111", "pickup_branch": "NOR"}],
      ["place_hold", {"patron_id": "P-217", "title_id": "BK-111", "pickup_branch": "NOR"}],
      ["cancel_hold", {"hold_id": "HD-550"}],
      ["place_hold", {"patron_id": "P-201", "title_id": "BK-111", "pickup_branch": "CEN"}],
      ["cancel_hold", {"hold_id": "HD-586"}], ["list_copies", {"title_id": "BK-111"}],
      ["list_holds", {"title_id": "BK-111"}],
      ["fill_hold", {"hold_id": "HD-600", "copy_id": "CP-1111"}],
      ["fill_hold", {"hold_id": "HD-601", "copy_id": "CP-1114"}]],
     [["cancel_hold", {"hold_id": "HD-507"}],
      ["place_hold", {"patron_id": "P-211", "title_id": "BK-111", "pickup_branch": "NOR"}],
      ["cancel_hold", {"hold_id": "HD-550"}],
      ["place_hold", {"patron_id": "P-217", "title_id": "BK-111", "pickup_branch": "NOR"}],
      ["cancel_hold", {"hold_id": "HD-586"}],
      ["place_hold", {"patron_id": "P-201", "title_id": "BK-111", "pickup_branch": "CEN"}],
      ["fill_hold", {"hold_id": "HD-600", "copy_id": "CP-1111"}],
      ["fill_hold", {"hold_id": "HD-601", "copy_id": "CP-1114"}]],
     _h03, "PRSBCNO", ref_errors=[5], mentions=["The Salt Road"], seq=[[6, 8, 12, 13], [7, 8]])


def _lib_h04():
    st = base._lib()
    st["holds"]["HD-587"] = _hold("P-202", "BK-150", "CEN", "ready", "09-10", "CP-1505", "2026-11-09")
    st["holds"]["HD-588"] = _hold("P-202", "BK-140", "CEN", "suspended", "10-20")
    st["holds"]["HD-589"] = _hold("P-202", "BK-112", "CEN", "in_transit", "10-01", "CP-1123")
    st["copies"]["CP-1505"] = _copy("BK-150", "CEN", "on_hold_shelf", hold="HD-587")
    st["copies"]["CP-1123"] = _copy("BK-112", "NOR", "in_transit", dest="CEN", hold="HD-589")
    return st


_add("H04", "library", _lib_h04(),
     "Ines Vargas and Ines Varga turn out to be the same person, and the Vargas record is being merged into the "
     "Varga one. Move every active hold on the Vargas record to Varga: Varga gets a hold on the same title with the "
     "same pickup branch (a suspended hold stays suspended) and the Vargas hold is cancelled. A hold whose copy is "
     "still travelling must have the copy checked in before the hold can be cancelled, and any copy freed this way "
     "goes to the next hold in line under the queue rules. Where Varga already has an active hold on the same title, "
     "keep whichever of the two is further along (ready, then in transit, then waiting or suspended; if equal, the "
     "earlier one) where it is, and cancel the other. If the rules don't let Varga hold a title at all, leave that "
     "Vargas hold untouched. Nothing else should change.",
     [["find_patrons", {"query": "Ines Varga"}], ["list_holds", {"patron_id": "P-202"}],
      ["list_holds", {"patron_id": "P-201"}], ["get_patron", {"patron_id": "P-201"}],
      ["place_hold", {"patron_id": "P-201", "title_id": "BK-120", "pickup_branch": "NOR"}],
      ["cancel_hold", {"hold_id": "HD-570"}],
      ["place_hold", {"patron_id": "P-201", "title_id": "BK-140", "pickup_branch": "CEN"}],
      ["set_hold_suspended", {"hold_id": "HD-600", "suspended": True}],
      ["cancel_hold", {"hold_id": "HD-588"}], ["receive_copy", {"copy_id": "CP-1123"}],
      ["place_hold", {"patron_id": "P-201", "title_id": "BK-112", "pickup_branch": "CEN"}],
      ["cancel_hold", {"hold_id": "HD-589"}], ["list_holds", {"title_id": "BK-112"}],
      ["fill_hold", {"hold_id": "HD-562", "copy_id": "CP-1123"}]],
     [["cancel_hold", {"hold_id": "HD-570"}],
      ["place_hold", {"patron_id": "P-201", "title_id": "BK-140", "pickup_branch": "CEN"}],
      ["set_hold_suspended", {"hold_id": "HD-600", "suspended": True}],
      ["cancel_hold", {"hold_id": "HD-588"}], ["receive_copy", {"copy_id": "CP-1123"}],
      ["place_hold", {"patron_id": "P-201", "title_id": "BK-112", "pickup_branch": "CEN"}],
      ["cancel_hold", {"hold_id": "HD-589"}], ["cancel_hold", {"hold_id": "HD-593"}],
      ["place_hold", {"patron_id": "P-201", "title_id": "BK-120", "pickup_branch": "NOR"}]],
     lambda s, i: _final(s, i, changes={
         "holds": {"HD-570": _CANC, "HD-588": _CANC, "HD-589": _CANC,
                   "HD-562": {"status": "in_transit", "copy": "CP-1123"}},
         "copies": {"CP-1123": {"location": "CEN", "destination": "NOR", "status": "in_transit", "hold": "HD-562"}}},
         new={"holds": [
             {"patron": "P-201", "title": "BK-140", "pickup": "CEN", "status": "suspended", "placed": "2026-11-04",
              "copy": "", "ready_until": ""},
             {"patron": "P-201", "title": "BK-112", "pickup": "CEN", "status": "waiting", "placed": "2026-11-04",
              "copy": "", "ready_until": ""}]}),
     "PRSBO", ref_errors=[4], mentions=["Ines Vargas", "Ines Varga"], seq=[[6, 7, 10], [9, 11, 13]])


def _lib_h05():
    st = base._lib()
    h, c = st["holds"], st["copies"]
    h["HD-598"] = _hold("P-209", "BK-150", "NOR", "ready", "10-15", "CP-1506", "2026-11-03")
    h["HD-599"] = _hold("P-203", "BK-121", "NOR", "ready", "10-25", "CP-1211", "2026-11-04")
    h["HD-578"] = _hold("P-217", "BK-161", "NOR", "ready", "10-20", "CP-1615", "2026-11-02")
    c["CP-1506"] = _copy("BK-150", "NOR", "on_hold_shelf", hold="HD-598")
    c["CP-1507"] = _copy("BK-150", "NOR", "on_loan", borrower="P-209", due="2026-11-20")
    c["CP-1615"] = _copy("BK-161", "NOR", "on_hold_shelf", hold="HD-578")
    c["CP-1211"]["status"], c["CP-1211"]["hold"] = "on_hold_shelf", "HD-599"
    return st


_add("H05", "library", _lib_h05(),
     "Please clear the expired holds off the Northgate hold shelf; a ready hold has expired once its last pickup day "
     "has passed. Cancel each expired one. If the patron then has fewer than three active holds, place a new hold for "
     "them on the same title for collection at Northgate - they go to the back of the queue - unless the rules don't "
     "allow it. After that, the freed copy goes to the next hold in line for its title under the queue rules; if "
     "nobody is eligible it stays on the shelf. Holds still inside their pickup window and other branches' hold "
     "shelves are not part of this, and nothing else should change.",
     [["get_library_info", {}], ["list_holds", {"pickup_branch": "NOR", "status": "ready"}],
      ["cancel_hold", {"hold_id": "HD-596"}], ["get_patron", {"patron_id": "P-206"}],
      ["place_hold", {"patron_id": "P-206", "title_id": "BK-112", "pickup_branch": "NOR"}],
      ["list_holds", {"title_id": "BK-112"}], ["fill_hold", {"hold_id": "HD-562", "copy_id": "CP-1121"}],
      ["cancel_hold", {"hold_id": "HD-598"}], ["get_patron", {"patron_id": "P-209"}],
      ["place_hold", {"patron_id": "P-209", "title_id": "BK-150", "pickup_branch": "NOR"}],
      ["list_holds", {"title_id": "BK-150"}], ["fill_hold", {"hold_id": "HD-551", "copy_id": "CP-1506"}],
      ["cancel_hold", {"hold_id": "HD-578"}], ["get_patron", {"patron_id": "P-217"}],
      ["list_holds", {"title_id": "BK-161"}]],
     [["cancel_hold", {"hold_id": "HD-596"}],
      ["place_hold", {"patron_id": "P-206", "title_id": "BK-112", "pickup_branch": "NOR"}],
      ["fill_hold", {"hold_id": "HD-562", "copy_id": "CP-1121"}], ["cancel_hold", {"hold_id": "HD-598"}],
      ["fill_hold", {"hold_id": "HD-551", "copy_id": "CP-1506"}], ["cancel_hold", {"hold_id": "HD-578"}],
      ["cancel_hold", {"hold_id": "HD-599"}],
      ["place_hold", {"patron_id": "P-203", "title_id": "BK-121", "pickup_branch": "NOR"}]],
     lambda s, i: _final(s, i, changes={
         "holds": {"HD-596": _CANC, "HD-598": _CANC, "HD-578": _CANC,
                   "HD-562": {"status": "ready", "copy": "CP-1121", "ready_until": _RDY},
                   "HD-551": {"status": "in_transit", "copy": "CP-1506"}},
         "copies": {"CP-1121": {"status": "on_hold_shelf", "hold": "HD-562"},
                    "CP-1506": {"status": "in_transit", "destination": "CEN", "hold": "HD-551"},
                    "CP-1615": {"status": "on_shelf", "hold": ""}}},
         new={"holds": [{"patron": "P-206", "title": "BK-112", "pickup": "NOR", "status": "waiting",
                         "placed": "2026-11-04", "copy": "", "ready_until": ""}]}),
     "PRSBC", ref_errors=[9], mentions=["Northgate"])


# --------------------------------------------------------------------------
# assets H06-H10
# --------------------------------------------------------------------------
_revoked = base._revoked
_newseat = base._newseat

_add("H06", "assets", base._assets(),
     "Bea Nunez (E-318) moves from Support to Engineering and Kenji Mori (E-317) from Engineering to Support, both "
     "today. For each of them: they lose every seat their new department is not allowed to hold, keep the ones it is "
     "allowed, and get a seat on every starter-bundle product of their new department that they don't hold yet. "
     "Seat counts must not be raised. Whenever you work on a licence, also take back every seat on it that is held "
     "by someone who has already left the company. Finally, any licence you worked on that ends up with more than "
     "one free seat has its count lowered so that exactly one is free. Devices don't change, and nothing else should "
     "change.",
     [["get_it_policy", {}], ["get_employee", {"employee_id": "E-318"}], ["get_employee", {"employee_id": "E-317"}],
      ["update_employee", {"employee_id": "E-318", "dept": "Engineering"}],
      ["revoke_seat", {"licence_id": "SW-41", "employee_id": "E-318"}],
      ["update_employee", {"employee_id": "E-318", "dept": "Engineering"}],
      ["revoke_seat", {"licence_id": "SW-31", "employee_id": "E-317"}],
      ["update_employee", {"employee_id": "E-317", "dept": "Support"}],
      ["list_seats", {"licence_id": "SW-31"}], ["revoke_seat", {"licence_id": "SW-31", "employee_id": "E-316"}],
      ["assign_seat", {"licence_id": "SW-31", "employee_id": "E-318"}],
      ["list_seats", {"licence_id": "SW-41"}], ["revoke_seat", {"licence_id": "SW-41", "employee_id": "E-335"}],
      ["assign_seat", {"licence_id": "SW-41", "employee_id": "E-317"}],
      ["list_seats", {"licence_id": "SW-22", "status": "active"}],
      ["list_seats", {"licence_id": "SW-22", "status": "active", "page": 2}],
      ["revoke_seat", {"licence_id": "SW-22", "employee_id": "E-328"}],
      ["assign_seat", {"licence_id": "SW-22", "employee_id": "E-317"}],
      ["list_licences", {}], ["set_seat_count", {"licence_id": "SW-31", "seats": 7}]],
     [["revoke_seat", {"licence_id": "SW-41", "employee_id": "E-318"}],
      ["update_employee", {"employee_id": "E-318", "dept": "Engineering"}],
      ["revoke_seat", {"licence_id": "SW-31", "employee_id": "E-317"}],
      ["revoke_seat", {"licence_id": "SW-32", "employee_id": "E-317"}],
      ["update_employee", {"employee_id": "E-317", "dept": "Support"}],
      ["assign_seat", {"licence_id": "SW-31", "employee_id": "E-318"}],
      ["assign_seat", {"licence_id": "SW-41", "employee_id": "E-317"}],
      ["assign_seat", {"licence_id": "SW-22", "employee_id": "E-317"}]],
     lambda s, i: _final(s, i, changes={
         "employees": {"E-318": {"dept": "Engineering"}, "E-317": {"dept": "Support"}},
         "seats": _revoked(i, [("SW-41", "E-318"), ("SW-31", "E-317"), ("SW-31", "E-316"), ("SW-41", "E-335"),
                               ("SW-22", "E-328")]),
         "licences": {"SW-31": {"seats": 7}}},
         new={"seats": [_newseat("SW-31", "E-318"), _newseat("SW-41", "E-317"), _newseat("SW-22", "E-317")]}),
     "PRSBCO", ref_errors=[3], mentions=["Bea Nunez", "Kenji Mori", "Support", "Engineering"])

_add("H07", "assets", base._assets(),
     "Three designers need software, in this priority order: Tariq Aziz, Noemi Laine, Theo Brandvold. Each needs a "
     "seat on every Design starter-bundle product they don't hold yet. For each product, cover them in priority "
     "order using, in this order of preference: free seats; seats taken back from people who have left the company; "
     "seats taken back from current staff whose seat has been inactive for longer than policy allows (longest "
     "inactive first); and finally a seat-count increase, but never more than one extra seat per product. Take back "
     "or add only as many seats as are actually needed. Whoever still can't be covered on a product simply doesn't "
     "get it. Nothing else should change.",
     [["get_it_policy", {}], ["find_employees", {"dept": "Design"}], ["get_employee", {"employee_id": "E-304"}],
      ["list_licences", {}], ["list_seats", {"licence_id": "SW-23"}],
      ["revoke_seat", {"licence_id": "SW-23", "employee_id": "E-307"}],
      ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-304"}],
      ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-340"}],
      ["set_seat_count", {"licence_id": "SW-23", "seats": 7}],
      ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-340"}],
      ["assign_seat", {"licence_id": "SW-11", "employee_id": "E-340"}],
      ["assign_seat", {"licence_id": "SW-11", "employee_id": "E-341"}],
      ["list_seats", {"licence_id": "SW-52"}], ["list_seats", {"licence_id": "SW-52", "page": 2}],
      ["assign_seat", {"licence_id": "SW-52", "employee_id": "E-304"}],
      ["set_seat_count", {"licence_id": "SW-52", "seats": 11}],
      ["assign_seat", {"licence_id": "SW-52", "employee_id": "E-340"}], ["list_licences", {}]],
     [["revoke_seat", {"licence_id": "SW-23", "employee_id": "E-307"}],
      ["set_seat_count", {"licence_id": "SW-23", "seats": 8}],
      ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-304"}],
      ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-340"}],
      ["assign_seat", {"licence_id": "SW-23", "employee_id": "E-341"}],
      ["assign_seat", {"licence_id": "SW-11", "employee_id": "E-340"}],
      ["assign_seat", {"licence_id": "SW-11", "employee_id": "E-341"}],
      ["set_seat_count", {"licence_id": "SW-52", "seats": 11}],
      ["assign_seat", {"licence_id": "SW-52", "employee_id": "E-304"}],
      ["assign_seat", {"licence_id": "SW-52", "employee_id": "E-340"}]],
     lambda s, i: _final(s, i, changes={
         "seats": _revoked(i, [("SW-23", "E-307")]),
         "licences": {"SW-23": {"seats": 7}, "SW-52": {"seats": 11}}},
         new={"seats": [_newseat("SW-23", "E-304"), _newseat("SW-23", "E-340"), _newseat("SW-11", "E-340"),
                        _newseat("SW-11", "E-341"), _newseat("SW-52", "E-304"), _newseat("SW-52", "E-340")]}),
     "PRSBCO", ref_errors=[7], mentions=["Tariq Aziz", "Noemi Laine", "Theo Brandvold", "Design"])


def _assets_h08():
    st = base._assets()
    st["devices"]["D-4029"] = base._dev_rec("laptop", "ThinkPad T14", "2021-02-02", "in_use", "E-316")
    return st


_add("H08", "assets", _assets_h08(),
     "Clean up after the people who left the company before 1 October 2026. Take back everything they still hold: "
     "devices go back into stock - except a laptop past the policy age limit, which is retired instead - and every "
     "software seat is revoked. Then, for each licence that lost a seat this way, set its seat count to the number "
     "of seats still in use plus one spare. People who left on or after 1 October, and anyone still working out "
     "their notice, are being handled separately: leave them and their seats alone. Nothing else should change.",
     [["get_it_policy", {}], ["find_employees", {"status": "left"}], ["get_employee", {"employee_id": "E-316"}],
      ["get_employee", {"employee_id": "E-328"}], ["retire_device", {"device_id": "D-4029"}],
      ["unassign_device", {"device_id": "D-4029", "destination": "in_stock"}],
      ["retire_device", {"device_id": "D-4029"}],
      ["revoke_seat", {"licence_id": "SW-12", "employee_id": "E-316"}],
      ["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-316"}],
      ["revoke_seat", {"licence_id": "SW-31", "employee_id": "E-316"}],
      ["revoke_seat", {"licence_id": "SW-22", "employee_id": "E-328"}], ["list_licences", {}],
      ["set_seat_count", {"licence_id": "SW-12", "seats": 3}], ["set_seat_count", {"licence_id": "SW-21", "seats": 9}],
      ["set_seat_count", {"licence_id": "SW-31", "seats": 7}], ["set_seat_count", {"licence_id": "SW-22", "seats": 11}]],
     [["unassign_device", {"device_id": "D-4029", "destination": "in_stock"}],
      ["revoke_seat", {"licence_id": "SW-12", "employee_id": "E-316"}],
      ["revoke_seat", {"licence_id": "SW-21", "employee_id": "E-316"}],
      ["revoke_seat", {"licence_id": "SW-31", "employee_id": "E-316"}],
      ["revoke_seat", {"licence_id": "SW-22", "employee_id": "E-328"}],
      ["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-307"}],
      ["revoke_seat", {"licence_id": "SW-23", "employee_id": "E-307"}],
      ["set_seat_count", {"licence_id": "SW-12", "seats": 3}], ["set_seat_count", {"licence_id": "SW-21", "seats": 9}],
      ["set_seat_count", {"licence_id": "SW-31", "seats": 7}], ["set_seat_count", {"licence_id": "SW-22", "seats": 11}]],
     lambda s, i: _final(s, i, changes={
         "devices": {"D-4029": {"status": "retired", "assigned_to": "", "last_user": "E-316"}},
         "seats": _revoked(i, [("SW-12", "E-316"), ("SW-21", "E-316"), ("SW-31", "E-316"), ("SW-22", "E-328")]),
         "licences": {"SW-12": {"seats": 3}, "SW-21": {"seats": 9}, "SW-31": {"seats": 7}, "SW-22": {"seats": 11}}}),
     "PRSBC", ref_errors=[4])

_add("H09", "assets", base._assets(),
     "D-4040 and D-4045 are back from repair today and go into stock. Then hand out laptops from stock in this "
     "priority: first everyone who is starting (lowest employee id first), then every active employee who has no "
     "laptop at all (lowest id first), then active employees whose laptop is past the policy age limit, oldest "
     "laptop first - their old laptop is retired once they have the new one. Each person takes the newest laptop "
     "that policy allows us to hand out from what is in stock at that moment. Stop when nothing suitable is left. "
     "People who are leaving are not part of this, and nothing else should change.",
     [["get_it_policy", {}], ["set_device_status", {"device_id": "D-4040", "status": "in_stock"}],
      ["set_device_status", {"device_id": "D-4045", "status": "in_stock"}],
      ["find_employees", {"status": "starting"}], ["list_devices", {"kind": "laptop", "status": "in_stock"}],
      ["assign_device", {"device_id": "D-4031", "employee_id": "E-340"}],
      ["assign_device", {"device_id": "D-4030", "employee_id": "E-340"}],
      ["assign_device", {"device_id": "D-4032", "employee_id": "E-341"}],
      ["assign_device", {"device_id": "D-4033", "employee_id": "E-342"}],
      ["find_employees", {"status": "active"}],
      ["list_devices", {"kind": "laptop", "status": "in_use"}],
      ["list_devices", {"kind": "laptop", "status": "in_use", "page": 2}],
      ["list_devices", {"kind": "laptop", "status": "in_use", "page": 3}],
      ["list_devices", {"kind": "laptop", "status": "in_use", "page": 4}],
      ["assign_device", {"device_id": "D-4040", "employee_id": "E-330"}],
      ["assign_device", {"device_id": "D-4045", "employee_id": "E-321"}],
      ["unassign_device", {"device_id": "D-4019", "destination": "in_stock"}],
      ["assign_device", {"device_id": "D-4045", "employee_id": "E-321"}],
      ["retire_device", {"device_id": "D-4019"}]],
     [["set_device_status", {"device_id": "D-4040", "status": "in_stock"}],
      ["set_device_status", {"device_id": "D-4045", "status": "in_stock"}],
      ["assign_device", {"device_id": "D-4030", "employee_id": "E-340"}],
      ["assign_device", {"device_id": "D-4032", "employee_id": "E-341"}],
      ["assign_device", {"device_id": "D-4033", "employee_id": "E-342"}],
      ["assign_device", {"device_id": "D-4040", "employee_id": "E-330"}],
      ["unassign_device", {"device_id": "D-4011", "destination": "in_stock"}],
      ["assign_device", {"device_id": "D-4045", "employee_id": "E-312"}],
      ["retire_device", {"device_id": "D-4011"}]],
     lambda s, i: _final(s, i, changes={"devices": {
         "D-4030": {"status": "in_use", "assigned_to": "E-340"}, "D-4032": {"status": "in_use", "assigned_to": "E-341"},
         "D-4033": {"status": "in_use", "assigned_to": "E-342"}, "D-4040": {"status": "in_use", "assigned_to": "E-330"},
         "D-4045": {"status": "in_use", "assigned_to": "E-321"},
         "D-4019": {"status": "retired", "assigned_to": "", "last_user": "E-321"}}}),
     "PRSBCO", ref_errors=[5, 15], seq=[[1, 14], [2, 17]])


def _assets_h10():
    st = base._assets()
    st["employees"]["E-337"] = {"name": "Bo Lindqvist", "dept": "Design", "status": "leaving", "end_date": "2026-12-01"}
    st["seats"]["SE-195"] = {"licence": "SW-11", "employee": "E-337", "assigned": "2025-01-06",
                             "last_used": "2026-11-10", "status": "active"}
    st["seats"]["SE-196"] = {"licence": "SW-52", "employee": "E-337", "assigned": "2025-01-06",
                             "last_used": "2026-11-10", "status": "active"}
    return st


_add("H10", "assets", _assets_h10(),
     "Right-size our licences without taking a seat from anyone. For every licence, treat the seats of people whose "
     "employment ends within the next 30 days (end date no more than 30 days after today) as already free. If a "
     "licence then has more than two free seats on that basis, lower its seat count so that exactly two would be "
     "free - but where our policy doesn't let the count go that low, set it to the lowest count policy allows "
     "instead. Don't revoke any seats, and change nothing else.",
     [["get_it_policy", {}], ["find_employees", {"status": "leaving"}], ["get_employee", {"employee_id": "E-309"}],
      ["get_employee", {"employee_id": "E-312"}], ["get_employee", {"employee_id": "E-337"}],
      ["list_licences", {}], ["set_seat_count", {"licence_id": "SW-11", "seats": 12}],
      ["list_seats", {"licence_id": "SW-11", "status": "active"}],
      ["set_seat_count", {"licence_id": "SW-11", "seats": 13}],
      ["set_seat_count", {"licence_id": "SW-12", "seats": 5}],
      ["set_seat_count", {"licence_id": "SW-32", "seats": 9}], ["list_licences", {}]],
     [["revoke_seat", {"licence_id": "SW-11", "employee_id": "E-337"}],
      ["set_seat_count", {"licence_id": "SW-11", "seats": 12}],
      ["set_seat_count", {"licence_id": "SW-12", "seats": 5}],
      ["set_seat_count", {"licence_id": "SW-32", "seats": 9}],
      ["set_seat_count", {"licence_id": "SW-52", "seats": 9}]],
     lambda s, i: _final(s, i, changes={"licences": {
         "SW-11": {"seats": 13}, "SW-12": {"seats": 5}, "SW-32": {"seats": 9}}}),
     "PRSCN", ref_errors=[6])


# --------------------------------------------------------------------------
# kitchen H11-H15
# --------------------------------------------------------------------------
_qty_after = base._qty_after


def _merge_alloc(*allocs):
    out = {}
    for a in allocs:
        for k, v in a.items():
            out[k] = out.get(k, 0) + v
    return out


_add("H11", "kitchen", base._kitchen(),
     "Chicken thigh is short this week. Allocate chicken thigh - and only chicken thigh - to every batch still "
     "planned for Tuesday 24 or Thursday 26 November that uses it, going through those batches by date and then by "
     "batch id and following the stock rules. A batch whose chicken can't be fully covered is cut to the largest "
     "whole number of portions that the chicken still available to it (counting what it already has) can cover, "
     "provided that is at least 40 portions; otherwise the batch is cancelled. Other ingredients will be dealt with "
     "later, and nothing else should change.",
     [["get_kitchen_info", {}], ["list_lots", {"ingredient_id": "ING-01"}],
      ["list_batches", {"date": "2026-11-24"}], ["list_batches", {"date": "2026-11-26"}],
      ["allocate_lot", {"batch_id": "PB-810", "lot_id": "LOT-0102", "grams": 12000}],
      ["allocate_lot", {"batch_id": "PB-810", "lot_id": "LOT-0101", "grams": 4000}],
      ["allocate_lot", {"batch_id": "PB-810", "lot_id": "LOT-0102", "grams": 8000}],
      ["allocate_lot", {"batch_id": "PB-820", "lot_id": "LOT-0102", "grams": 13000}],
      ["allocate_lot", {"batch_id": "PB-820", "lot_id": "LOT-0103", "grams": 6000}],
      ["allocate_lot", {"batch_id": "PB-820", "lot_id": "LOT-0104", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-821", "lot_id": "LOT-0104", "grams": 12000}],
      ["list_lots", {"ingredient_id": "ING-01"}], ["get_batch", {"batch_id": "PB-802"}],
      ["update_batch", {"batch_id": "PB-802", "portions": 41}],
      ["allocate_lot", {"batch_id": "PB-802", "lot_id": "LOT-0104", "grams": 2920}],
      ["cancel_batch", {"batch_id": "PB-805"}], ["cancel_batch", {"batch_id": "PB-851"}]],
     [["allocate_lot", {"batch_id": "PB-810", "lot_id": "LOT-0101", "grams": 4000}],
      ["allocate_lot", {"batch_id": "PB-810", "lot_id": "LOT-0102", "grams": 8000}],
      ["allocate_lot", {"batch_id": "PB-820", "lot_id": "LOT-0102", "grams": 13000}],
      ["allocate_lot", {"batch_id": "PB-820", "lot_id": "LOT-0103", "grams": 6000}],
      ["allocate_lot", {"batch_id": "PB-820", "lot_id": "LOT-0104", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-821", "lot_id": "LOT-0104", "grams": 12000}],
      ["cancel_batch", {"batch_id": "PB-802"}], ["cancel_batch", {"batch_id": "PB-805"}],
      ["cancel_batch", {"batch_id": "PB-851"}]],
     lambda s, i: _final(s, i, changes={"batches": {
         "PB-810": {"alloc": {"LOT-0101": 4000, "LOT-0102": 8000}},
         "PB-820": {"alloc": {"LOT-0102": 13000, "LOT-0103": 6000, "LOT-0104": 5000}},
         "PB-821": {"alloc": {"LOT-0401": 1000, "LOT-0104": 12000}},
         "PB-802": {"portions": 41, "alloc": {"LOT-0103": 2000, "LOT-0104": 2920}},
         "PB-805": {"status": "cancelled", "alloc": {}}, "PB-851": {"status": "cancelled", "alloc": {}}}}),
     "PRSBCO", ref_errors=[4], mentions=["Chicken thigh"], seq=[[5, 6, 7, 8, 9, 10, 14]])


def _kitchen_h12():
    st = base._kitchen()
    b = st["batches"]
    b["PB-854"]["alloc"] = {"LOT-1202": 13500, "LOT-1301": 2000, "LOT-1303": 10000, "LOT-1402": 2250}
    b["PB-852"]["alloc"] = {"LOT-2101": 10800, "LOT-0501": 2700, "LOT-0402": 8100}
    b["PB-853"]["alloc"] = {"LOT-0101": 4000}
    b["PB-820"]["alloc"] = {"LOT-0102": 21000, "LOT-0103": 3000, "LOT-0202": 20000, "LOT-0301": 4000,
                            "LOT-0403": 18000}
    b["PB-814"]["alloc"] = {"LOT-2201": 350, "LOT-0605": 500, "LOT-0601": 550, "LOT-2401": 7000}
    b["PB-810"]["alloc"] = {"LOT-0104": 12000, "LOT-0501": 3000, "LOT-0403": 9000}
    return st


_H12_852 = {"LOT-2101": 10800, "LOT-0501": 2700, "LOT-0401": 5000, "LOT-0403": 3100}
_H12_854 = {"LOT-1202": 13500, "LOT-1301": 2000, "LOT-1303": 10000, "LOT-1402": 2250}
_H12_820 = {"LOT-0102": 21000, "LOT-0103": 3000, "LOT-0202": 20000, "LOT-0301": 4000, "LOT-0403": 18000}
_H12_814 = {"LOT-2201": 350, "LOT-0601": 1050, "LOT-2401": 7000}

_add("H12", "kitchen", _kitchen_h12(),
     "Release to production every batch planned for today (Monday 23 November) or tomorrow that is already fully "
     "allocated and is allowed to run on the line it is planned on. Before releasing a batch, check its allocations: "
     "stock taken from a lot that is no longer usable for that batch must be taken off and replaced from usable stock "
     "following the stock rules. Don't allocate anything for batches that aren't fully allocated yet, and leave a "
     "batch that may not run where it is planned exactly as it is. Nothing else should change.",
     [["get_kitchen_info", {}], ["list_batches", {"date": "2026-11-23"}], ["list_batches", {"date": "2026-11-24"}],
      ["get_batch", {"batch_id": "PB-854"}], ["release_batch", {"batch_id": "PB-854"}],
      ["get_batch", {"batch_id": "PB-852"}], ["deallocate_lot", {"batch_id": "PB-852", "lot_id": "LOT-0402"}],
      ["allocate_lot", {"batch_id": "PB-852", "lot_id": "LOT-0403", "grams": 8100}],
      ["allocate_lot", {"batch_id": "PB-852", "lot_id": "LOT-0401", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-852", "lot_id": "LOT-0403", "grams": 3100}],
      ["release_batch", {"batch_id": "PB-852"}], ["get_batch", {"batch_id": "PB-820"}],
      ["release_batch", {"batch_id": "PB-820"}], ["get_batch", {"batch_id": "PB-814"}],
      ["deallocate_lot", {"batch_id": "PB-814", "lot_id": "LOT-0605"}],
      ["allocate_lot", {"batch_id": "PB-814", "lot_id": "LOT-0601", "grams": 500}],
      ["release_batch", {"batch_id": "PB-814"}], ["get_batch", {"batch_id": "PB-810"}],
      ["get_recipe", {"recipe_id": "RC-03"}]],
     [["release_batch", {"batch_id": "PB-854"}], ["release_batch", {"batch_id": "PB-852"}],
      ["release_batch", {"batch_id": "PB-820"}], ["release_batch", {"batch_id": "PB-814"}],
      ["release_batch", {"batch_id": "PB-810"}]],
     lambda s, i: _final(s, i, changes={
         "batches": {"PB-854": {"status": "released"}, "PB-852": {"status": "released", "alloc": _H12_852},
                     "PB-820": {"status": "released"}, "PB-814": {"status": "released", "alloc": _H12_814}},
         "lots": _qty_after(i, _merge_alloc(_H12_854, _H12_852, _H12_820, _H12_814))}),
     "PRSBN", ref_errors=[7], seq=[[6, 8, 9, 10], [14, 15, 16]])

_H13_THU = {"LOT-0103": 6000, "LOT-0104": 1200, "LOT-1601": 1500, "LOT-0401": 5000, "LOT-0403": 400, "LOT-2201": 180}
_H13_FRI = {"LOT-0104": 7200, "LOT-1601": 1500, "LOT-0403": 5400, "LOT-2201": 180}

_add("H13", "kitchen", base._kitchen(),
     "Half of Friday's Teriyaki Chicken batch has to be made a day earlier. Plan a new Teriyaki Chicken batch with "
     "half of its portions on Thursday 26 November on the same line, and reduce Friday's batch to the other half. "
     "Allocate every ingredient of the Thursday batch first and then of the Friday batch, following the stock rules, "
     "and release the Thursday batch to production; Friday's batch stays planned. Nothing else should change.",
     [["get_kitchen_info", {}], ["get_batch", {"batch_id": "PB-841"}],
      ["create_batch", {"recipe_id": "RC-05", "portions": 60, "date": "2026-11-26", "line": "L1"}],
      ["update_batch", {"batch_id": "PB-841", "portions": 60}],
      ["list_lots", {"ingredient_id": "ING-01"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0103", "grams": 6000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0104", "grams": 1200}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1601", "grams": 1500}],
      ["list_lots", {"ingredient_id": "ING-04"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0401", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0403", "grams": 400}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-2201", "grams": 180}],
      ["release_batch", {"batch_id": "PB-900"}],
      ["allocate_lot", {"batch_id": "PB-841", "lot_id": "LOT-0104", "grams": 7200}],
      ["allocate_lot", {"batch_id": "PB-841", "lot_id": "LOT-1601", "grams": 1500}],
      ["allocate_lot", {"batch_id": "PB-841", "lot_id": "LOT-0401", "grams": 5400}],
      ["allocate_lot", {"batch_id": "PB-841", "lot_id": "LOT-0403", "grams": 5400}],
      ["allocate_lot", {"batch_id": "PB-841", "lot_id": "LOT-2201", "grams": 180}]],
     [["create_batch", {"recipe_id": "RC-05", "portions": 60, "date": "2026-11-26", "line": "L1"}],
      ["update_batch", {"batch_id": "PB-841", "portions": 60}],
      ["allocate_lot", {"batch_id": "PB-841", "lot_id": "LOT-0103", "grams": 6000}],
      ["allocate_lot", {"batch_id": "PB-841", "lot_id": "LOT-0104", "grams": 1200}],
      ["allocate_lot", {"batch_id": "PB-841", "lot_id": "LOT-1601", "grams": 1500}],
      ["allocate_lot", {"batch_id": "PB-841", "lot_id": "LOT-0403", "grams": 5400}],
      ["allocate_lot", {"batch_id": "PB-841", "lot_id": "LOT-2201", "grams": 180}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0104", "grams": 7200}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1601", "grams": 1500}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0401", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-0403", "grams": 400}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-2201", "grams": 180}],
      ["release_batch", {"batch_id": "PB-900"}]],
     lambda s, i: _final(s, i, changes={
         "batches": {"PB-841": {"portions": 60, "alloc": _H13_FRI}}, "lots": _qty_after(i, _H13_THU)},
         new={"batches": [{"recipe": "RC-05", "portions": 60, "date": "2026-11-26", "line": "L1", "status": "released",
                           "alloc": _H13_THU}]}),
     "PRSCO", ref_errors=[15], mentions=["Teriyaki Chicken"],
     seq=[[2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17]])

_add("H14", "kitchen", base._kitchen(),
     "Fen Growers has recalled every lot of theirs that expires before 1 December 2026. Quarantine all of those lots "
     "with reason recall. Every planned batch that had stock from them needs it replaced from usable lots following "
     "the stock rules, dealing with the batches in order of batch id. A batch that can't be fully re-supplied is cut "
     "to the largest number of portions that can be fully covered - with any other allocation that is now too big "
     "taken off and allocated again for the new size - and if that comes to zero portions, the batch is cancelled "
     "instead. Nothing else should change.",
     [["get_kitchen_info", {}], ["list_lots", {"expires_before": "2026-12-01"}],
      ["list_lots", {"expires_before": "2026-12-01", "page": 2}],
      ["quarantine_lot", {"lot_id": "LOT-1302", "reason": "recall"}],
      ["get_batch", {"batch_id": "PB-830"}], ["get_batch", {"batch_id": "PB-832"}],
      ["list_lots", {"ingredient_id": "ING-13"}],
      ["cancel_batch", {"batch_id": "PB-830"}], ["cancel_batch", {"batch_id": "PB-832"}],
      ["get_batch", {"batch_id": "PB-844"}],
      ["deallocate_lot", {"batch_id": "PB-844", "lot_id": "LOT-1001"}],
      ["quarantine_lot", {"lot_id": "LOT-1001", "reason": "recall"}],
      ["quarantine_lot", {"lot_id": "LOT-1002", "reason": "recall"}],
      ["quarantine_lot", {"lot_id": "LOT-1301", "reason": "recall"}],
      ["quarantine_lot", {"lot_id": "LOT-1302", "reason": "recall"}],
      ["quarantine_lot", {"lot_id": "LOT-1303", "reason": "recall"}],
      ["allocate_lot", {"batch_id": "PB-844", "lot_id": "LOT-1003", "grams": 4000}]],
     [["deallocate_lot", {"batch_id": "PB-830", "lot_id": "LOT-1302"}],
      ["deallocate_lot", {"batch_id": "PB-832", "lot_id": "LOT-1302"}],
      ["deallocate_lot", {"batch_id": "PB-844", "lot_id": "LOT-1001"}],
      ["quarantine_lot", {"lot_id": "LOT-1001", "reason": "recall"}],
      ["quarantine_lot", {"lot_id": "LOT-1002", "reason": "recall"}],
      ["quarantine_lot", {"lot_id": "LOT-1003", "reason": "recall"}],
      ["quarantine_lot", {"lot_id": "LOT-1301", "reason": "recall"}],
      ["quarantine_lot", {"lot_id": "LOT-1302", "reason": "recall"}],
      ["quarantine_lot", {"lot_id": "LOT-1303", "reason": "recall"}]],
     lambda s, i: _final(s, i, changes={
         "lots": dict((l, {"status": "quarantined", "reason": "recall"})
                      for l in ("LOT-1001", "LOT-1002", "LOT-1301", "LOT-1302", "LOT-1303")),
         "batches": {"PB-830": {"status": "cancelled", "alloc": {}}, "PB-832": {"status": "cancelled", "alloc": {}},
                     "PB-844": {"alloc": {"LOT-1003": 4000}}}}),
     "PRSBCO", ref_errors=[3], mentions=["Fen Growers"], seq=[[10, 11, 16]])

_H15 = {"LOT-1202": 13500, "LOT-1303": 12000, "LOT-1401": 1000, "LOT-1402": 1250}

_add("H15", "kitchen", base._kitchen(),
     "A client wants 150 portions of Mushroom Risotto made on Line 1 on Thursday 26 November, fully allocated and "
     "released today. If Line 1 doesn't have room for it that day, move exactly one planned batch off Line 1 to make "
     "room: the smallest batch whose move frees enough room, to whichever other line it is allowed on that has the "
     "most room. Then plan the client's batch, allocate it following the stock rules and release it. If usable stock "
     "can't cover all 150 portions, don't plan it at all. Nothing else should change.",
     [["get_kitchen_info", {}], ["list_batches", {"date": "2026-11-26"}],
      ["create_batch", {"recipe_id": "RC-09", "portions": 150, "date": "2026-11-26", "line": "L1"}],
      ["update_batch", {"batch_id": "PB-802", "line": "L3"}],
      ["get_recipe", {"recipe_id": "RC-09"}], ["list_lots", {"ingredient_id": "ING-12"}],
      ["list_lots", {"ingredient_id": "ING-13"}], ["list_lots", {"ingredient_id": "ING-14"}],
      ["create_batch", {"recipe_id": "RC-09", "portions": 150, "date": "2026-11-26", "line": "L1"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1202", "grams": 13500}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1303", "grams": 12000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1401", "grams": 1000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1402", "grams": 1250}],
      ["release_batch", {"batch_id": "PB-900"}]],
     [["update_batch", {"batch_id": "PB-802", "line": "L2"}],
      ["create_batch", {"recipe_id": "RC-09", "portions": 150, "date": "2026-11-26", "line": "L3"}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1202", "grams": 13500}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1303", "grams": 12000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1401", "grams": 1000}],
      ["allocate_lot", {"batch_id": "PB-900", "lot_id": "LOT-1402", "grams": 1250}],
      ["release_batch", {"batch_id": "PB-900"}]],
     lambda s, i: _final(s, i, changes={"batches": {"PB-802": {"line": "L3"}}, "lots": _qty_after(i, _H15)},
                         new={"batches": [{"recipe": "RC-09", "portions": 150, "date": "2026-11-26", "line": "L1",
                                           "status": "released", "alloc": _H15}]}),
     "PRSCO", ref_errors=[2], mentions=["Mushroom Risotto", "Line 1"], seq=[[3, 8, 9, 10, 11, 12, 13]])


# --------------------------------------------------------------------------
# moderation H16-H20
# --------------------------------------------------------------------------
_post = base._post_rec
_rep = base._rep_rec
_user = base._user_rec


def _mod_h16():
    st = base._mod()
    st["posts"]["PO-2075"] = _post("U-131", "c/kochen", "11-28", "visible", "Nachtgedanken")
    st["posts"]["PO-2076"] = _post("U-130", "c/kochen", "12-01", "visible", "kurzer Beitrag")
    st["reports"]["RP-3120"] = _rep("PO-2075", "U-155", "self_harm", "11-29")
    st["reports"]["RP-3121"] = _rep("PO-2033", "U-156", "threat", "11-30", assignee="M-02")
    st["reports"]["RP-3122"] = _rep("PO-2076", "U-157", "self_harm", "12-01")
    return st


_add("H16", "moderation", _mod_h16(),
     "Every open self_harm or threat report has to be escalated to a tier 2 or tier 3 moderator. Go through the "
     "open ones (escalated ones are already fine) oldest first, and escalate each to the tier 2 or 3 moderator who "
     "works in the community's language and has the fewest open reports at that moment (ties: lower id) - including "
     "reports a tier 1 moderator is currently holding. If no such moderator has capacity, first make room: take the "
     "eligible tier 2 or 3 moderator with the fewest open reports (ties: lower id) and move their most recently "
     "filed open report that a tier 1 moderator may hold to the tier 1 moderator in that report's language with the "
     "fewest open reports (ties: lower id). Nothing else should change.",
     [["get_mod_policy", {}], ["list_reports", {"status": "open", "reason": "self_harm"}],
      ["list_reports", {"status": "open", "reason": "threat"}], ["list_moderators", {}],
      ["escalate_report", {"report_id": "RP-3046", "moderator_id": "M-08"}],
      ["escalate_report", {"report_id": "RP-3044", "moderator_id": "M-08"}],
      ["escalate_report", {"report_id": "RP-3120", "moderator_id": "M-05"}],
      ["escalate_report", {"report_id": "RP-3121", "moderator_id": "M-08"}],
      ["escalate_report", {"report_id": "RP-3122", "moderator_id": "M-05"}],
      ["list_moderators", {}], ["list_reports", {"assignee": "M-05"}],
      ["assign_report", {"report_id": "RP-3094", "moderator_id": "M-07"}],
      ["escalate_report", {"report_id": "RP-3122", "moderator_id": "M-05"}]],
     [["escalate_report", {"report_id": "RP-3046", "moderator_id": "M-08"}],
      ["escalate_report", {"report_id": "RP-3044", "moderator_id": "M-08"}],
      ["escalate_report", {"report_id": "RP-3120", "moderator_id": "M-05"}],
      ["escalate_report", {"report_id": "RP-3121", "moderator_id": "M-08"}],
      ["assign_report", {"report_id": "RP-3030", "moderator_id": "M-07"}],
      ["escalate_report", {"report_id": "RP-3122", "moderator_id": "M-04"}]],
     lambda s, i: _final(s, i, changes={"reports": {
         "RP-3046": {"status": "escalated", "assignee": "M-08"}, "RP-3044": {"status": "escalated", "assignee": "M-08"},
         "RP-3120": {"status": "escalated", "assignee": "M-05"}, "RP-3121": {"status": "escalated", "assignee": "M-08"},
         "RP-3122": {"status": "escalated", "assignee": "M-05"}, "RP-3094": {"assignee": "M-07"}}}),
     "PRSBCO", ref_errors=[8], seq=[[4, 5, 7], [6, 8, 11, 12]])


def _mod_h17():
    st = base._mod()
    st["posts"]["PO-2007"] = _post("U-101", "c/retro-games", "11-29", "visible", "rant about a streamer")
    st["posts"]["PO-2008"] = _post("U-101", "c/retro-games", "11-30", "visible", "follow-up rant")
    st["reports"]["RP-3123"] = _rep("PO-2007", "U-157", "hate", "11-30")
    st["reports"]["RP-3124"] = _rep("PO-2008", "U-158", "hate", "12-01")
    return st


_add("H17", "moderation", _mod_h17(),
     "All open hate reports in c/local-news and c/retro-games have been confirmed. Resolve each one as actioned, "
     "remove the reported post if it isn't removed already, and give the post's author one strike for each removed "
     "post. Then apply our strike thresholds to every author involved: a suspension is signed off by Chen (M-03), "
     "and anything that needs a more senior sign-off goes to the eligible moderator with the fewest open reports "
     "(ties: lower id). Hate reports in other communities, and reports for other reasons, are not part of this.",
     [["get_mod_policy", {}], ["list_reports", {"status": "open", "reason": "hate", "community": "c/local-news"}],
      ["list_reports", {"status": "open", "reason": "hate", "community": "c/retro-games"}],
      ["resolve_report", {"report_id": "RP-3050", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3051", "outcome": "actioned"}],
      ["set_post_status", {"post_id": "PO-2040", "status": "removed"}],
      ["add_strike", {"user_id": "U-140", "post_id": "PO-2040"}],
      ["resolve_report", {"report_id": "RP-3123", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3124", "outcome": "actioned"}],
      ["set_post_status", {"post_id": "PO-2007", "status": "removed"}],
      ["set_post_status", {"post_id": "PO-2008", "status": "removed"}],
      ["add_strike", {"user_id": "U-101", "post_id": "PO-2007"}],
      ["add_strike", {"user_id": "U-101", "post_id": "PO-2008"}],
      ["get_user", {"user_id": "U-140"}], ["get_user", {"user_id": "U-101"}],
      ["set_user_status", {"user_id": "U-140", "status": "banned", "approved_by": "M-03"}],
      ["list_moderators", {}],
      ["set_user_status", {"user_id": "U-140", "status": "banned", "approved_by": "M-05"}],
      ["set_user_status", {"user_id": "U-101", "status": "suspended", "approved_by": "M-03"}]],
     [["resolve_report", {"report_id": "RP-3050", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3051", "outcome": "actioned"}],
      ["set_post_status", {"post_id": "PO-2040", "status": "removed"}],
      ["add_strike", {"user_id": "U-140", "post_id": "PO-2040"}],
      ["resolve_report", {"report_id": "RP-3123", "outcome": "actioned"}],
      ["resolve_report", {"report_id": "RP-3124", "outcome": "actioned"}],
      ["set_post_status", {"post_id": "PO-2007", "status": "removed"}],
      ["set_post_status", {"post_id": "PO-2008", "status": "removed"}],
      ["add_strike", {"user_id": "U-101", "post_id": "PO-2007"}],
      ["add_strike", {"user_id": "U-101", "post_id": "PO-2008"}],
      ["set_user_status", {"user_id": "U-140", "status": "suspended", "approved_by": "M-03"}],
      ["set_user_status", {"user_id": "U-101", "status": "suspended", "approved_by": "M-03"}]],
     lambda s, i: _final(s, i, changes={
         "reports": dict((r, {"status": "actioned"}) for r in ("RP-3050", "RP-3051", "RP-3123", "RP-3124")),
         "posts": dict((p, {"status": "removed"}) for p in ("PO-2040", "PO-2007", "PO-2008")),
         "users": {"U-140": {"strike_posts": ["PO-1940", "PO-1941", "PO-1942", "PO-1943", "PO-2040"],
                             "status": "banned"},
                   "U-101": {"strike_posts": base._set(["PO-1901", "PO-1902", "PO-2007", "PO-2008"]), "status": "suspended",
                             "suspended_until": "2026-12-09"}}}),
     "PRSBCO", ref_errors=[15], mentions=["c/local-news", "c/retro-games", "Chen"],
     seq=[[5, 6, 17], [9, 11], [10, 12], [11, 18], [12, 18]])


def _mod_h18():
    st = base._mod()
    u = st["users"]["U-140"]
    u["status"], u["strike_posts"] = "banned", ["PO-1940", "PO-1941", "PO-1942", "PO-1943", "PO-2042"]
    st["posts"]["PO-2042"] = _post("U-140", "c/local-news", "11-20", "removed", "slur")
    return st


_add("H18", "moderation", _mod_h18(),
     "These appeals have been upheld: PO-2010, PO-1943, PO-2034 and PO-2015. For each of those posts, put it back "
     "up, take back the strike its author was given for it, and dismiss every report on it that is still open or "
     "escalated (reports that were already actioned stay as they are). Afterwards, every author whose account is "
     "restricted - suspended or banned - and who has dropped below the threshold for that restriction gets it "
     "lifted, signed off by Dagmar (M-04), wherever policy allows that. Nothing else should change.",
     [["get_mod_policy", {}], ["get_post", {"post_id": "PO-2010"}], ["get_post", {"post_id": "PO-1943"}],
      ["get_post", {"post_id": "PO-2034"}], ["get_post", {"post_id": "PO-2015"}],
      ["set_post_status", {"post_id": "PO-2010", "status": "visible"}],
      ["remove_strike", {"user_id": "U-110", "post_id": "PO-2010"}],
      ["resolve_report", {"report_id": "RP-3010", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3011", "outcome": "dismissed"}],
      ["set_post_status", {"post_id": "PO-1943", "status": "visible"}],
      ["remove_strike", {"user_id": "U-140", "post_id": "PO-1943"}],
      ["set_post_status", {"post_id": "PO-2034", "status": "visible"}],
      ["remove_strike", {"user_id": "U-134", "post_id": "PO-2034"}],
      ["set_post_status", {"post_id": "PO-2015", "status": "visible"}],
      ["remove_strike", {"user_id": "U-111", "post_id": "PO-2015"}],
      ["get_user", {"user_id": "U-110"}], ["get_user", {"user_id": "U-140"}],
      ["set_user_status", {"user_id": "U-140", "status": "active", "approved_by": "M-04"}],
      ["set_user_status", {"user_id": "U-110", "status": "active", "approved_by": "M-04"}]],
     [["set_post_status", {"post_id": "PO-2010", "status": "visible"}],
      ["remove_strike", {"user_id": "U-110", "post_id": "PO-2010"}],
      ["resolve_report", {"report_id": "RP-3010", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3011", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3012", "outcome": "dismissed"}],
      ["set_post_status", {"post_id": "PO-1943", "status": "visible"}],
      ["remove_strike", {"user_id": "U-140", "post_id": "PO-1943"}],
      ["set_post_status", {"post_id": "PO-2034", "status": "visible"}],
      ["remove_strike", {"user_id": "U-134", "post_id": "PO-2034"}],
      ["set_post_status", {"post_id": "PO-2015", "status": "visible"}],
      ["remove_strike", {"user_id": "U-111", "post_id": "PO-2015"}]],
     lambda s, i: _final(s, i, changes={
         "posts": dict((p, {"status": "visible"}) for p in ("PO-2010", "PO-1943", "PO-2034", "PO-2015")),
         "reports": {"RP-3010": {"status": "dismissed"}, "RP-3011": {"status": "dismissed"}},
         "users": {"U-110": {"strike_posts": ["PO-2011", "PO-2012"], "status": "active", "suspended_until": ""},
                   "U-140": {"strike_posts": ["PO-1940", "PO-1941", "PO-1942", "PO-2042"]},
                   "U-134": {"strike_posts": []}, "U-111": {"strike_posts": []}}}),
     "PRSBO", ref_errors=[17], mentions=["Dagmar"], seq=[[6, 18]])


def _mod_h19():
    st = base._mod()
    st["users"]["U-194"] = _user("fresh_sprout", "2026-11-28")
    st["users"]["U-195"] = _user("leafbot_99", "2026-11-30")
    u = st["users"]["U-190"]
    u["strike_posts"], u["status"], u["suspended_until"] = ["PO-2100", "PO-2103", "PO-2104"], "suspended", "2026-12-05"
    st["users"]["U-163"]["strike_posts"] = ["PO-2101"]
    p = st["posts"]
    p["PO-2100"] = _post("U-190", "c/gardening", "11-27", "removed", "tomato seedlings for swap")
    p["PO-2101"] = _post("U-163", "c/gardening", "11-28", "removed", "argument about peat")
    p["PO-2102"] = _post("U-125", "c/gardening", "11-29", "hidden", "photos of my allotment")
    p["PO-2103"] = _post("U-190", "c/gardening", "10-01", "removed", "spam link")
    p["PO-2104"] = _post("U-190", "c/parenting", "10-15", "removed", "spam link")
    r = st["reports"]
    r["RP-3126"] = _rep("PO-2100", "U-194", "spam", "11-28", "actioned", "M-01")
    r["RP-3127"] = _rep("PO-2100", "U-195", "spam", "11-30", "actioned", "M-01")
    r["RP-3128"] = _rep("PO-2101", "U-194", "harassment", "11-29", "actioned", "M-03")
    r["RP-3129"] = _rep("PO-2101", "U-152", "harassment", "11-29", "actioned", "M-03")
    r["RP-3130"] = _rep("PO-2102", "U-195", "spam", "12-01")
    r["RP-3131"] = _rep("PO-2060", "U-194", "spam", "12-01")
    return st


_add("H19", "moderation", _mod_h19(),
     "@fresh_sprout and @leafbot_99 were created this week only to brigade c/gardening. Undo the damage. Dismiss "
     "every report of theirs that is still open. A post that was removed only because of their reports - every "
     "actioned report on it came from those two accounts - is put back up, and its author loses the strike given "
     "for it. A hidden post whose only open reports came from them is made visible. Posts where someone else's "
     "report was also actioned stay exactly as they are. Finally, any author who is suspended and has dropped below "
     "the suspension threshold is reinstated, signed off by Chen (M-03). Nothing else should change.",
     [["get_mod_policy", {}], ["find_users", {"query": "fresh_sprout"}], ["find_users", {"query": "leafbot_99"}],
      ["list_reports", {"reporter_id": "U-194"}], ["list_reports", {"reporter_id": "U-195"}],
      ["resolve_report", {"report_id": "RP-3130", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3131", "outcome": "dismissed"}],
      ["get_post", {"post_id": "PO-2100"}], ["get_post", {"post_id": "PO-2101"}],
      ["get_post", {"post_id": "PO-2102"}],
      ["set_post_status", {"post_id": "PO-2100", "status": "visible"}],
      ["set_user_status", {"user_id": "U-190", "status": "active", "approved_by": "M-03"}],
      ["remove_strike", {"user_id": "U-190", "post_id": "PO-2100"}],
      ["set_post_status", {"post_id": "PO-2102", "status": "visible"}],
      ["get_user", {"user_id": "U-190"}],
      ["set_user_status", {"user_id": "U-190", "status": "active", "approved_by": "M-03"}]],
     [["resolve_report", {"report_id": "RP-3130", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3131", "outcome": "dismissed"}],
      ["set_post_status", {"post_id": "PO-2100", "status": "visible"}],
      ["remove_strike", {"user_id": "U-190", "post_id": "PO-2100"}],
      ["set_post_status", {"post_id": "PO-2101", "status": "visible"}],
      ["remove_strike", {"user_id": "U-163", "post_id": "PO-2101"}],
      ["set_post_status", {"post_id": "PO-2102", "status": "visible"}],
      ["set_user_status", {"user_id": "U-190", "status": "active", "approved_by": "M-03"}]],
     lambda s, i: _final(s, i, changes={
         "reports": {"RP-3130": {"status": "dismissed"}, "RP-3131": {"status": "dismissed"}},
         "posts": {"PO-2100": {"status": "visible"}, "PO-2102": {"status": "visible"}},
         "users": {"U-190": {"strike_posts": ["PO-2103", "PO-2104"], "status": "active", "suspended_until": ""}}}),
     "PRSBO", ref_errors=[11], mentions=["fresh_sprout", "leafbot_99", "c/gardening", "Chen"], seq=[[12, 15]])


def _mod_h20():
    st = base._mod()
    st["reports"]["RP-3132"] = _rep("PO-2003", "U-159", "harassment", "11-22", "escalated", "M-03")
    st["reports"]["RP-3133"] = _rep("PO-2020", "U-150", "spam", "11-25", "escalated", "M-08")
    st["reports"]["RP-3134"] = _rep("PO-2061", "U-151", "hate", "11-20", "escalated", "M-05")
    st["users"]["U-184"]["strike_posts"] = ["PO-1985", "PO-1986"]
    st["posts"]["PO-1985"] = _post("U-184", "c/futbol", "10-05", "removed", "amenaza")
    st["posts"]["PO-1986"] = _post("U-184", "c/futbol", "11-02", "removed", "insulto")
    return st


_add("H20", "moderation", _mod_h20(),
     "Please work through the escalated reports that were filed more than five days ago. If the reported post is "
     "hidden, remove it, resolve the report as actioned and give the post's author a strike for it - suspending the "
     "author if that brings them to the threshold, with Hector (M-08) signing off. If the post is still visible, "
     "dismiss the report, unless policy doesn't allow that report to be dismissed, in which case it simply stays "
     "escalated as it is. Escalated reports from the last five days are not part of this, and nothing else should "
     "change.",
     [["get_mod_policy", {}], ["list_reports", {"status": "escalated"}],
      ["get_post", {"post_id": "PO-2031"}],
      ["resolve_report", {"report_id": "RP-3031", "outcome": "dismissed"}],
      ["get_post", {"post_id": "PO-2087"}], ["set_post_status", {"post_id": "PO-2087", "status": "removed"}],
      ["resolve_report", {"report_id": "RP-3087", "outcome": "actioned"}],
      ["add_strike", {"user_id": "U-184", "post_id": "PO-2087"}], ["get_user", {"user_id": "U-184"}],
      ["set_user_status", {"user_id": "U-184", "status": "suspended", "approved_by": "M-08"}],
      ["get_post", {"post_id": "PO-2003"}], ["resolve_report", {"report_id": "RP-3132", "outcome": "dismissed"}],
      ["get_post", {"post_id": "PO-2020"}], ["resolve_report", {"report_id": "RP-3133", "outcome": "dismissed"}],
      ["get_post", {"post_id": "PO-2061"}], ["resolve_report", {"report_id": "RP-3134", "outcome": "dismissed"}]],
     [["set_post_status", {"post_id": "PO-2087", "status": "removed"}],
      ["resolve_report", {"report_id": "RP-3087", "outcome": "actioned"}],
      ["add_strike", {"user_id": "U-184", "post_id": "PO-2087"}],
      ["set_user_status", {"user_id": "U-184", "status": "suspended", "approved_by": "M-08"}],
      ["resolve_report", {"report_id": "RP-3132", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3133", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3134", "outcome": "dismissed"}],
      ["resolve_report", {"report_id": "RP-3093", "outcome": "dismissed"}]],
     lambda s, i: _final(s, i, changes={
         "posts": {"PO-2087": {"status": "removed"}},
         "reports": {"RP-3087": {"status": "actioned"}, "RP-3132": {"status": "dismissed"},
                     "RP-3133": {"status": "dismissed"}, "RP-3134": {"status": "dismissed"}},
         "users": {"U-184": {"strike_posts": ["PO-1985", "PO-1986", "PO-2087"], "status": "suspended",
                             "suspended_until": "2026-12-09"}}}),
     "PRSBC", ref_errors=[3], mentions=["Hector"], seq=[[5, 7, 9]])


# --------------------------------------------------------------------------
# exams H21-H25
# --------------------------------------------------------------------------
_gone = base._gone
_asg = base._asg

_add("H21", "exams", base._exams(),
     "Kit Byrne and Joan Pell will both be away on Wednesday 13 and Thursday 14 January. Take them off every exam "
     "they invigilate on those two days and record both of them as unavailable on both dates. Replace them in the "
     "same role, handling the affected exams in order of date, start time and exam id: each time pick the eligible "
     "member of staff with the fewest sessions that day at that point (ties: lower staff id). Their exams on other "
     "days stay as they are, and nothing else should change.",
     [["get_exam_policy", {}], ["find_staff", {"query": "Kit Byrne"}], ["find_staff", {"query": "Joan Pell"}],
      ["list_staff_schedule", {"date": "2027-01-13"}], ["list_staff_schedule", {"date": "2027-01-14"}],
      ["set_staff_unavailable", {"staff_id": "ST-16", "date": "2027-01-13"}],
      ["remove_invigilator", {"exam_id": "EX-121", "staff_id": "ST-16"}],
      ["remove_invigilator", {"exam_id": "EX-123", "staff_id": "ST-01"}],
      ["remove_invigilator", {"exam_id": "EX-133", "staff_id": "ST-16"}],
      ["remove_invigilator", {"exam_id": "EX-134", "staff_id": "ST-01"}],
      ["set_staff_unavailable", {"staff_id": "ST-16", "date": "2027-01-13"}],
      ["set_staff_unavailable", {"staff_id": "ST-16", "date": "2027-01-14"}],
      ["set_staff_unavailable", {"staff_id": "ST-01", "date": "2027-01-13"}],
      ["set_staff_unavailable", {"staff_id": "ST-01", "date": "2027-01-14"}],
      ["assign_invigilator", {"exam_id": "EX-121", "staff_id": "ST-11", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-123", "staff_id": "ST-06", "role": "support"}],
      ["list_staff_schedule", {"date": "2027-01-14"}],
      ["assign_invigilator", {"exam_id": "EX-133", "staff_id": "ST-05", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-134", "staff_id": "ST-06", "role": "support"}]],
     [["remove_invigilator", {"exam_id": "EX-121", "staff_id": "ST-16"}],
      ["remove_invigilator", {"exam_id": "EX-123", "staff_id": "ST-01"}],
      ["remove_invigilator", {"exam_id": "EX-133", "staff_id": "ST-16"}],
      ["remove_invigilator", {"exam_id": "EX-134", "staff_id": "ST-01"}],
      ["set_staff_unavailable", {"staff_id": "ST-16", "date": "2027-01-13"}],
      ["set_staff_unavailable", {"staff_id": "ST-16", "date": "2027-01-14"}],
      ["set_staff_unavailable", {"staff_id": "ST-01", "date": "2027-01-13"}],
      ["set_staff_unavailable", {"staff_id": "ST-01", "date": "2027-01-14"}],
      ["assign_invigilator", {"exam_id": "EX-121", "staff_id": "ST-11", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-123", "staff_id": "ST-06", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-134", "staff_id": "ST-05", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-133", "staff_id": "ST-21", "role": "lead"}]],
     lambda s, i: _final(s, i, changes={
         "assignments": _gone(i, [("EX-121", "ST-16"), ("EX-123", "ST-01"), ("EX-133", "ST-16"), ("EX-134", "ST-01")]),
         "staff": {"ST-16": {"unavailable": ["2027-01-13", "2027-01-14"]},
                   "ST-01": {"unavailable": ["2027-01-13", "2027-01-14", "2027-01-15"]}}},
         new={"assignments": [_asg("EX-121", "ST-11", "lead"), _asg("EX-123", "ST-06", "support"),
                              _asg("EX-133", "ST-05", "lead"), _asg("EX-134", "ST-06", "support")]}),
     "PRSCO", ref_errors=[5], mentions=["Kit Byrne", "Joan Pell"], seq=[[17, 18]])

_add("H22", "exams", base._exams(),
     "Two Thursday-morning papers, Physics Paper 1 and then Economics Paper 1, may move to Friday 15 January at the "
     "same start time - but only if the exam can be fully staffed there. Handle them in that order. A moved exam "
     "keeps its room if that room is free then, otherwise it goes to the smallest free room that seats everyone. "
     "Its current invigilators who can still do Friday keep their roles; anyone who can't is replaced by an "
     "eligible member of staff in the same role with the fewest sessions that day (ties: lower staff id). If an "
     "exam can't get a complete set of invigilators, including a qualified lead, on Friday, it stays exactly where "
     "it is with its current invigilators. Nothing else should change.",
     [["get_exam_policy", {}], ["get_exam", {"exam_id": "EX-131"}], ["list_exams", {"date": "2027-01-15"}],
      ["list_staff_schedule", {"date": "2027-01-15"}],
      ["reschedule_exam", {"exam_id": "EX-131", "date": "2027-01-15", "start": "09:00", "room_id": "RM-02"}],
      ["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-14"}],
      ["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-03"}],
      ["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-13"}],
      ["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-20"}],
      ["reschedule_exam", {"exam_id": "EX-131", "date": "2027-01-15", "start": "09:00", "room_id": "RM-02"}],
      ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-16", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-03", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-13", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-20", "role": "support"}],
      ["get_exam", {"exam_id": "EX-132"}], ["list_staff_schedule", {"date": "2027-01-15"}],
      ["find_staff", {"lead_qualified": True}]],
     [["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-14"}],
      ["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-03"}],
      ["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-13"}],
      ["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-20"}],
      ["reschedule_exam", {"exam_id": "EX-131", "date": "2027-01-15", "start": "09:00", "room_id": "RM-02"}],
      ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-16", "role": "lead"}],
      ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-03", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-13", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-20", "role": "support"}],
      ["remove_invigilator", {"exam_id": "EX-132", "staff_id": "ST-02"}],
      ["remove_invigilator", {"exam_id": "EX-132", "staff_id": "ST-07"}],
      ["reschedule_exam", {"exam_id": "EX-132", "date": "2027-01-15", "start": "09:00", "room_id": "RM-03"}]],
     lambda s, i: _final(s, i, changes={
         "exams": {"EX-131": {"date": "2027-01-15"}},
         "assignments": _gone(i, [("EX-131", "ST-14"), ("EX-131", "ST-03"), ("EX-131", "ST-13"), ("EX-131", "ST-20")])},
         new={"assignments": [_asg("EX-131", "ST-16", "lead"), _asg("EX-131", "ST-03", "support"),
                              _asg("EX-131", "ST-13", "support"), _asg("EX-131", "ST-20", "support")]}),
     "PRSCNO", ref_errors=[4], mentions=["Physics Paper 1", "Economics Paper 1"],
     seq=[[5, 9], [6, 9], [7, 9], [8, 9], [9, 10], [9, 11], [9, 12], [9, 13]])

_add("H23", "exams", base._exams(),
     "Late entries have come in for three Friday papers: German Paper 1 now has 54 candidates, Spanish Paper 1 has "
     "65 and Physics Paper 3 has 64. Deal with them in order of start time, then exam id. Each exam must sit in a "
     "room that seats everyone: it keeps its room if that fits, otherwise it moves to the smallest room that seats "
     "everyone and is free at that time. It must then have exactly the number of invigilators policy requires - add "
     "support invigilators, choosing free non-teaching staff in order of staff id. If an exam can't be given a big "
     "enough room, leave its candidate number and everything else about it unchanged. Nothing else should change.",
     [["get_exam_policy", {}], ["list_exams", {"date": "2027-01-15"}], ["list_rooms", {}],
      ["list_staff_schedule", {"date": "2027-01-15"}],
      ["set_exam_candidates", {"exam_id": "EX-141", "candidates": 54}],
      ["assign_invigilator", {"exam_id": "EX-141", "staff_id": "ST-03", "role": "support"}],
      ["set_exam_candidates", {"exam_id": "EX-143", "candidates": 64}],
      ["set_exam_room", {"exam_id": "EX-143", "room_id": "RM-02"}],
      ["set_exam_candidates", {"exam_id": "EX-143", "candidates": 64}],
      ["get_exam", {"exam_id": "EX-143"}],
      ["assign_invigilator", {"exam_id": "EX-143", "staff_id": "ST-04", "role": "support"}],
      ["get_exam", {"exam_id": "EX-144"}], ["list_exams", {"date": "2027-01-15"}]],
     [["set_exam_candidates", {"exam_id": "EX-141", "candidates": 54}],
      ["assign_invigilator", {"exam_id": "EX-141", "staff_id": "ST-03", "role": "support"}],
      ["set_exam_room", {"exam_id": "EX-144", "room_id": "RM-02"}],
      ["set_exam_candidates", {"exam_id": "EX-144", "candidates": 65}],
      ["assign_invigilator", {"exam_id": "EX-144", "staff_id": "ST-04", "role": "support"}]],
     lambda s, i: _final(s, i, changes={
         "exams": {"EX-141": {"candidates": 54}, "EX-143": {"candidates": 64, "room": "RM-02"}}},
         new={"assignments": [_asg("EX-141", "ST-03", "support"), _asg("EX-143", "ST-04", "support")]}),
     "PRSCO", ref_errors=[6], mentions=["German Paper 1", "Spanish Paper 1", "Physics Paper 3"],
     seq=[[4, 5], [7, 8, 10]])

_add("H24", "exams", base._exams(),
     "Hugo Varga and Leo Park are swapping all of their invigilation duties next week: each takes over every exam "
     "session the other is down for, in the same role, and gives up his own. Where both of them are on the same "
     "exam, nothing changes for that exam. Every exam should end up with the same number of invigilators it has "
     "now. Rita Varga is not involved, and nothing else should change.",
     [["find_staff", {}], ["list_exams", {}], ["list_exams", {"page": 2}], ["list_exams", {"page": 3}],
      ["assign_invigilator", {"exam_id": "EX-112", "staff_id": "ST-20", "role": "support"}],
      ["remove_invigilator", {"exam_id": "EX-112", "staff_id": "ST-22"}],
      ["remove_invigilator", {"exam_id": "EX-114", "staff_id": "ST-20"}],
      ["assign_invigilator", {"exam_id": "EX-112", "staff_id": "ST-20", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-114", "staff_id": "ST-22", "role": "support"}],
      ["remove_invigilator", {"exam_id": "EX-121", "staff_id": "ST-20"}],
      ["remove_invigilator", {"exam_id": "EX-123", "staff_id": "ST-22"}],
      ["assign_invigilator", {"exam_id": "EX-121", "staff_id": "ST-22", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-123", "staff_id": "ST-20", "role": "support"}],
      ["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-20"}],
      ["remove_invigilator", {"exam_id": "EX-134", "staff_id": "ST-22"}],
      ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-22", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-134", "staff_id": "ST-20", "role": "support"}],
      ["remove_invigilator", {"exam_id": "EX-143", "staff_id": "ST-20"}],
      ["assign_invigilator", {"exam_id": "EX-143", "staff_id": "ST-22", "role": "support"}],
      ["list_staff_schedule", {"date": "2027-01-15"}]],
     [["remove_invigilator", {"exam_id": "EX-103", "staff_id": "ST-20"}],
      ["remove_invigilator", {"exam_id": "EX-103", "staff_id": "ST-22"}],
      ["remove_invigilator", {"exam_id": "EX-112", "staff_id": "ST-22"}],
      ["remove_invigilator", {"exam_id": "EX-114", "staff_id": "ST-20"}],
      ["assign_invigilator", {"exam_id": "EX-112", "staff_id": "ST-20", "role": "support"}],
      ["assign_invigilator", {"exam_id": "EX-114", "staff_id": "ST-22", "role": "support"}]],
     lambda s, i: _final(s, i, changes={
         "assignments": _gone(i, [("EX-114", "ST-20"), ("EX-121", "ST-20"), ("EX-131", "ST-20"), ("EX-143", "ST-20"),
                                  ("EX-112", "ST-22"), ("EX-123", "ST-22"), ("EX-134", "ST-22")])},
         new={"assignments": [_asg("EX-112", "ST-20", "support"), _asg("EX-114", "ST-22", "support"),
                              _asg("EX-123", "ST-20", "support"), _asg("EX-121", "ST-22", "support"),
                              _asg("EX-134", "ST-20", "support"), _asg("EX-131", "ST-22", "support"),
                              _asg("EX-143", "ST-22", "support")]}),
     "PRSBO", ref_errors=[4], mentions=["Hugo Varga", "Leo Park", "Rita Varga"])

_add("H25", "exams", base._exams(),
     "Spread next week's invigilation more fairly. Everyone with five or more sessions next week gives up exactly "
     "one session; handle them in order of staff id. For each of them, try their sessions from the latest one (by "
     "date and start time) backwards and take the first session where someone can take over in the same role: an "
     "eligible member of staff who is not already on that exam and has at most three sessions in the whole week - "
     "the one with the fewest sessions in the week (ties: lower staff id). Take the busy person off that session and "
     "put the replacement in. Session counts are as they stand at each point. Nothing else should change.",
     [["get_exam_policy", {}], ["list_exams", {}], ["list_exams", {"page": 2}], ["list_exams", {"page": 3}],
      ["list_staff_schedule", {"date": "2027-01-15"}], ["list_staff_schedule", {"date": "2027-01-14"}],
      ["assign_invigilator", {"exam_id": "EX-132", "staff_id": "ST-05", "role": "lead"}],
      ["remove_invigilator", {"exam_id": "EX-132", "staff_id": "ST-02"}],
      ["assign_invigilator", {"exam_id": "EX-132", "staff_id": "ST-05", "role": "lead"}],
      ["remove_invigilator", {"exam_id": "EX-142", "staff_id": "ST-03"}],
      ["assign_invigilator", {"exam_id": "EX-142", "staff_id": "ST-06", "role": "support"}],
      ["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-14"}],
      ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-09", "role": "lead"}],
      ["list_staff_schedule", {"date": "2027-01-14"}],
      ["remove_invigilator", {"exam_id": "EX-133", "staff_id": "ST-16"}],
      ["assign_invigilator", {"exam_id": "EX-133", "staff_id": "ST-05", "role": "lead"}],
      ["remove_invigilator", {"exam_id": "EX-143", "staff_id": "ST-20"}],
      ["assign_invigilator", {"exam_id": "EX-143", "staff_id": "ST-10", "role": "support"}]],
     [["remove_invigilator", {"exam_id": "EX-141", "staff_id": "ST-02"}],
      ["assign_invigilator", {"exam_id": "EX-141", "staff_id": "ST-16", "role": "lead"}],
      ["remove_invigilator", {"exam_id": "EX-142", "staff_id": "ST-03"}],
      ["assign_invigilator", {"exam_id": "EX-142", "staff_id": "ST-06", "role": "support"}],
      ["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-14"}],
      ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-09", "role": "lead"}],
      ["remove_invigilator", {"exam_id": "EX-143", "staff_id": "ST-20"}],
      ["assign_invigilator", {"exam_id": "EX-143", "staff_id": "ST-10", "role": "support"}]],
     lambda s, i: _final(s, i, changes={
         "assignments": _gone(i, [("EX-132", "ST-02"), ("EX-142", "ST-03"), ("EX-131", "ST-14"), ("EX-133", "ST-16"),
                                  ("EX-143", "ST-20")])},
         new={"assignments": [_asg("EX-132", "ST-05", "lead"), _asg("EX-142", "ST-06", "support"),
                              _asg("EX-131", "ST-09", "lead"), _asg("EX-133", "ST-05", "lead"),
                              _asg("EX-143", "ST-10", "support")]}),
     "PRSCO", ref_errors=[6], seq=[[7, 8], [9, 10], [11, 12], [14, 15], [16, 17]])


# --------------------------------------------------------------------------
# calibration HP1-HP3 (same difficulty; library, kitchen, exams)
# --------------------------------------------------------------------------
def _lib_hp1():
    st = base._lib()
    st["holds"]["HD-576"] = _hold("P-215", "BK-161", "CEN", "suspended", "10-20")
    st["holds"]["HD-577"] = _hold("P-215", "BK-150", "CEN", "in_transit", "09-12", "CP-1505")
    st["copies"]["CP-1505"] = _copy("BK-150", "NOR", "in_transit", dest="CEN", hold="HD-577")
    return st


_add("HP1", "library", _lib_hp1(),
     "Sofia Marin now collects everything at Riverside. Move each of her active holds there. Where the rules won't "
     "let a hold simply be redirected, check in a travelling copy at its destination first if needed, then cancel "
     "the hold and place it again for Riverside (she accepts going to the back of the queue). Suspended holds must "
     "still be suspended at the end. Once her holds are sorted, every copy freed along the way that is still on the "
     "open shelf goes to the next hold in line under the queue rules. Nothing else should change.",
     [["find_patrons", {"query": "Sofia Marin"}], ["list_holds", {"patron_id": "P-215"}],
      ["update_hold", {"hold_id": "HD-506", "pickup_branch": "RIV"}],
      ["update_hold", {"hold_id": "HD-590", "pickup_branch": "RIV"}],
      ["cancel_hold", {"hold_id": "HD-590"}],
      ["place_hold", {"patron_id": "P-215", "title_id": "BK-140", "pickup_branch": "RIV"}],
      ["set_hold_suspended", {"hold_id": "HD-576", "suspended": False}],
      ["update_hold", {"hold_id": "HD-576", "pickup_branch": "RIV"}],
      ["set_hold_suspended", {"hold_id": "HD-576", "suspended": True}],
      ["receive_copy", {"copy_id": "CP-1505"}], ["cancel_hold", {"hold_id": "HD-577"}],
      ["place_hold", {"patron_id": "P-215", "title_id": "BK-150", "pickup_branch": "RIV"}],
      ["list_holds", {"title_id": "BK-140"}], ["fill_hold", {"hold_id": "HD-533", "copy_id": "CP-1401"}],
      ["list_holds", {"title_id": "BK-150"}], ["fill_hold", {"hold_id": "HD-551", "copy_id": "CP-1505"}]],
     [["update_hold", {"hold_id": "HD-506", "pickup_branch": "RIV"}],
      ["cancel_hold", {"hold_id": "HD-590"}],
      ["place_hold", {"patron_id": "P-215", "title_id": "BK-140", "pickup_branch": "RIV"}],
      ["set_hold_suspended", {"hold_id": "HD-576", "suspended": False}],
      ["update_hold", {"hold_id": "HD-576", "pickup_branch": "RIV"}],
      ["set_hold_suspended", {"hold_id": "HD-576", "suspended": True}],
      ["receive_copy", {"copy_id": "CP-1505"}], ["cancel_hold", {"hold_id": "HD-577"}],
      ["place_hold", {"patron_id": "P-215", "title_id": "BK-150", "pickup_branch": "RIV"}],
      ["fill_hold", {"hold_id": "HD-602", "copy_id": "CP-1505"}]],
     lambda s, i: _final(s, i, changes={
         "holds": {"HD-506": {"pickup": "RIV"}, "HD-576": {"pickup": "RIV"}, "HD-590": _CANC, "HD-577": _CANC,
                   "HD-533": {"status": "ready", "copy": "CP-1401", "ready_until": _RDY},
                   "HD-551": {"status": "ready", "copy": "CP-1505", "ready_until": _RDY}},
         "copies": {"CP-1401": {"status": "on_hold_shelf", "hold": "HD-533"},
                    "CP-1505": {"location": "CEN", "destination": "", "status": "on_hold_shelf", "hold": "HD-551"}}},
         new={"holds": [
             {"patron": "P-215", "title": "BK-140", "pickup": "RIV", "status": "waiting", "placed": "2026-11-04",
              "copy": "", "ready_until": ""},
             {"patron": "P-215", "title": "BK-150", "pickup": "RIV", "status": "waiting", "placed": "2026-11-04",
              "copy": "", "ready_until": ""}]}),
     "PRSCO", ref_errors=[3], mentions=["Sofia Marin", "Riverside"], seq=[[4, 5, 13], [9, 10, 11, 15]])

_add("HP2", "kitchen", base._kitchen(),
     "Onions are short this week. Allocate onion - and only onion - to every batch planned for Tuesday 24, "
     "Wednesday 25 or Thursday 26 November that uses it, going through them by date and then batch id and following "
     "the stock rules. A batch whose onion can't be fully covered is cut to the largest whole number of portions the "
     "onion still available to it can cover, provided that is at least 40 portions; otherwise the batch is "
     "cancelled. Other ingredients will be dealt with later, and nothing else should change.",
     [["get_kitchen_info", {}], ["list_lots", {"ingredient_id": "ING-10"}],
      ["list_batches", {"date": "2026-11-24"}], ["list_batches", {"date": "2026-11-25"}],
      ["list_batches", {"date": "2026-11-26"}],
      ["allocate_lot", {"batch_id": "PB-822", "lot_id": "LOT-1003", "grams": 7500}],
      ["allocate_lot", {"batch_id": "PB-822", "lot_id": "LOT-1002", "grams": 2500}],
      ["allocate_lot", {"batch_id": "PB-822", "lot_id": "LOT-1003", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-831", "lot_id": "LOT-1003", "grams": 12500}],
      ["list_lots", {"ingredient_id": "ING-10"}], ["get_batch", {"batch_id": "PB-804"}],
      ["update_batch", {"batch_id": "PB-804", "portions": 62}],
      ["allocate_lot", {"batch_id": "PB-804", "lot_id": "LOT-1003", "grams": 2480}]],
     [["allocate_lot", {"batch_id": "PB-822", "lot_id": "LOT-1002", "grams": 2500}],
      ["allocate_lot", {"batch_id": "PB-822", "lot_id": "LOT-1003", "grams": 5000}],
      ["allocate_lot", {"batch_id": "PB-831", "lot_id": "LOT-1003", "grams": 12500}],
      ["update_batch", {"batch_id": "PB-804", "portions": 63}],
      ["allocate_lot", {"batch_id": "PB-804", "lot_id": "LOT-1003", "grams": 2500}]],
     lambda s, i: _final(s, i, changes={"batches": {
         "PB-822": {"alloc": {"LOT-1002": 2500, "LOT-1003": 5000}}, "PB-831": {"alloc": {"LOT-1003": 12500}},
         "PB-804": {"portions": 62, "alloc": {"LOT-1003": 2480}}}}),
     "PRSBCO", ref_errors=[5])

_add("HP3", "exams", base._exams(),
     "Gwen Tate can't invigilate on Monday 11 or Friday 15 January. For each of those days, take her off her exams "
     "and record her as unavailable that day. Replace her in each exam in the same role with an eligible member of "
     "staff who is not already on that exam and has the fewest sessions that day (ties: lower staff id). If one of "
     "her exams on a day can't be covered, she stays on that exam and is not recorded as unavailable that day, but "
     "the exams that can be covered that day are still handed over. Nothing else should change.",
     [["get_exam_policy", {}], ["find_staff", {"query": "Gwen Tate"}],
      ["list_staff_schedule", {"date": "2027-01-11"}],
      ["remove_invigilator", {"exam_id": "EX-101", "staff_id": "ST-14"}],
      ["set_staff_unavailable", {"staff_id": "ST-14", "date": "2027-01-11"}],
      ["find_staff", {"lead_qualified": True}],
      ["assign_invigilator", {"exam_id": "EX-101", "staff_id": "ST-07", "role": "lead"}],
      ["list_staff_schedule", {"date": "2027-01-15"}],
      ["remove_invigilator", {"exam_id": "EX-140", "staff_id": "ST-14"}],
      ["assign_invigilator", {"exam_id": "EX-140", "staff_id": "ST-16", "role": "lead"}],
      ["set_staff_unavailable", {"staff_id": "ST-14", "date": "2027-01-15"}],
      ["list_staff_schedule", {"date": "2027-01-15"}]],
     [["remove_invigilator", {"exam_id": "EX-101", "staff_id": "ST-14"}],
      ["set_staff_unavailable", {"staff_id": "ST-14", "date": "2027-01-11"}],
      ["assign_invigilator", {"exam_id": "EX-101", "staff_id": "ST-07", "role": "lead"}],
      ["remove_invigilator", {"exam_id": "EX-140", "staff_id": "ST-14"}],
      ["assign_invigilator", {"exam_id": "EX-140", "staff_id": "ST-16", "role": "lead"}],
      ["remove_invigilator", {"exam_id": "EX-144", "staff_id": "ST-14"}],
      ["set_staff_unavailable", {"staff_id": "ST-14", "date": "2027-01-15"}]],
     lambda s, i: _final(s, i, changes={
         "assignments": _gone(i, [("EX-101", "ST-14"), ("EX-140", "ST-14")]),
         "staff": {"ST-14": {"unavailable": ["2027-01-11"]}}},
         new={"assignments": [_asg("EX-101", "ST-07", "lead"), _asg("EX-140", "ST-16", "lead")]}),
     "PRSCO", ref_errors=[10], mentions=["Gwen Tate"], seq=[[3, 4], [3, 6], [8, 9]])


_ALT_ROUTES_H = {
    "H11": [["allocate_lot", {"batch_id": "PB-810", "lot_id": "LOT-0101", "grams": 4000}],
            ["allocate_lot", {"batch_id": "PB-810", "lot_id": "LOT-0102", "grams": 8000}],
            ["allocate_lot", {"batch_id": "PB-820", "lot_id": "LOT-0102", "grams": 13000}],
            ["allocate_lot", {"batch_id": "PB-820", "lot_id": "LOT-0103", "grams": 6000}],
            ["allocate_lot", {"batch_id": "PB-820", "lot_id": "LOT-0104", "grams": 5000}],
            ["allocate_lot", {"batch_id": "PB-821", "lot_id": "LOT-0104", "grams": 12000}],
            ["allocate_lot", {"batch_id": "PB-802", "lot_id": "LOT-0104", "grams": 3000}],
            ["update_batch", {"batch_id": "PB-802", "portions": 41}],
            ["deallocate_lot", {"batch_id": "PB-802", "lot_id": "LOT-0104"}],
            ["update_batch", {"batch_id": "PB-802", "portions": 41}],
            ["allocate_lot", {"batch_id": "PB-802", "lot_id": "LOT-0104", "grams": 2920}],
            ["cancel_batch", {"batch_id": "PB-851"}], ["cancel_batch", {"batch_id": "PB-805"}]],
    "H22": [["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-20"}],
            ["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-13"}],
            ["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-03"}],
            ["remove_invigilator", {"exam_id": "EX-131", "staff_id": "ST-14"}],
            ["reschedule_exam", {"exam_id": "EX-131", "date": "2027-01-15", "start": "09:00", "room_id": "RM-02"}],
            ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-03", "role": "support"}],
            ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-16", "role": "lead"}],
            ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-20", "role": "support"}],
            ["assign_invigilator", {"exam_id": "EX-131", "staff_id": "ST-13", "role": "support"}]],
}


# --------------------------------------------------------------------------
# self-verification
# --------------------------------------------------------------------------
_GARBAGE = base._GARBAGE


def _fuzz(tasks):
    rng = random.Random(20261023)
    calls = 0
    for ename in sorted(ENVS):
        cls = ENVS[ename]
        inits = [t["init"] for t in tasks if t["env"] == ename]
        if not inits:
            continue
        pool = set()
        for ini in inits:
            base._strings(ini, pool)
        pool = sorted(pool)
        dates = [x for x in pool if re.match(r"^\d{4}-\d{2}-\d{2}$", x)] or ["2026-11-04"]
        names = [t["function"]["name"] for t in cls.TOOLS]
        env = cls(inits[0])
        for name in names + [None, 5, "nope", "", [], {}, 1.5, "fill_hold; drop"]:
            for args in _GARBAGE + ["{}", 7]:
                base._safe_call(env, name, args, ename)
                calls += 1

        def rand_value(schema, depth=0):
            if rng.random() < 0.25 or depth > 3:
                return rng.choice(_GARBAGE)
            t = schema.get("type")
            if t == "string":
                if "enum" in schema and rng.random() < 0.85:
                    return rng.choice(schema["enum"])
                return rng.choice(pool)
            if t == "integer":
                return rng.choice([-5, 0, 1, 2, 3, 6, 8, 9, 10, 12, 41, 92, 100, 150, 1000, 2920, 9200])
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
            if "enum" in schema:
                return rng.choice(schema["enum"])
            t = schema.get("type")
            if t == "boolean":
                return rng.random() < 0.5
            if t == "integer":
                return rng.choice([1, 2, 3, 5, 8, 9, 10, 12, 20, 50, 200, 1000, 4000])
            if "date" in key or key == "expires_before":
                return rng.choice(dates)
            if key == "start":
                return rng.choice(["09:00", "13:30", "11:00"])
            return rng.choice(pool)

        ok_writes = 0
        for n in range(2000):
            if n % 100 == 0:
                env = cls(rng.choice(inits))
            tool = rng.choice(cls.TOOLS)["function"]
            if n % 2:
                args = rand_value(tool["parameters"])
            else:
                props = tool["parameters"]["properties"]
                args = dict((k, typed_value(k, sub)) for k, sub in props.items()
                            if k in tool["parameters"]["required"] or rng.random() < 0.3)
            res = base._safe_call(env, tool["name"], args, ename)
            if not _is_err(res) and not _is_read(tool["name"]):
                ok_writes += 1
            calls += 1
        assert ok_writes > 0, (ename, "random fuzz never produced a successful write")
    for t in tasks:
        state = ENVS[t["env"]](t["init"])
        for step, (name, args) in enumerate(t["reference"]):
            for key in list(args) + ["__extra__"]:
                for g in _GARBAGE:
                    env = ENVS[t["env"]](state.state)
                    bad = copy.deepcopy(args)
                    bad[key] = g
                    base._safe_call(env, name, bad, t["id"])
                    calls += 1
            for key in args:
                env = ENVS[t["env"]](state.state)
                bad = copy.deepcopy(args)
                del bad[key]
                base._safe_call(env, name, bad, t["id"])
                calls += 1
            state.call(name, copy.deepcopy(args))
    return calls


_ALT_ROUTES = dict(_ALT_ROUTES_H)


def _verify(expect_ids):
    ids = [t["id"] for t in TASKS]
    assert ids == expect_ids, ids
    by_id = dict((t["id"], t) for t in TASKS)
    ok = 0
    rng = random.Random(11)
    report = []
    for t in TASKS:
        tid = t["id"]
        cls = ENVS[t["env"]]
        tools = dict((x["function"]["name"], x["function"]) for x in cls.TOOLS)
        words = len(t["prompt"].split())
        assert 60 <= words <= 220, (tid, "prompt words", words)
        assert 12 <= len(t["reference"]) <= 20, (tid, "reference length", len(t["reference"]))
        f = set(t["features"])
        assert f <= set("PRSBCNO") and len(f) >= 4, (tid, "features", t["features"])
        assert {"P", "R", "S"} <= f and (f & set("BC")), (tid, "must combine P, R, S and B or C", t["features"])
        assert t["ref_errors"], (tid, "R: the reference must contain the natural rule error")
        json.dumps(t["init"])
        json.dumps(t["reference"])
        json.dumps(t["trap"])
        base._no_empty(t["reference"], tid)
        n_ent = base._prompt_entities(t)
        env = cls(t["init"])
        assert env.state == t["init"] and env.state is not t["init"]
        assert t["check"](env.state) is False, (tid, "check passes on the initial state")
        chars = 0
        first_write, reread = None, False
        for n, (name, args) in enumerate(t["reference"]):
            assert name in tools, (tid, "unknown tool in reference", name)
            before = copy.deepcopy(env.state)
            res = env.call(name, copy.deepcopy(args))
            chars += len(json.dumps(res))
            assert _is_err(res) == (n in t["ref_errors"]), (tid, "reference step", n, name, res)
            if _is_read(name) or _is_err(res):
                assert env.state == before, (tid, "state changed by read/error step", n)
                if _is_read(name) and first_write is not None:
                    reread = True
            else:
                assert env.state != before, (tid, "reference write changed nothing", n, name)
                if first_write is None:
                    first_write = n
        assert reread, (tid, "S: the reference never re-reads state after a write")
        assert t["check"](env.state) is True, (tid, "reference does not satisfy check")
        assert chars < 60000, (tid, "reference transcript too long", chars)
        for fn in tools.values():
            if not fn["parameters"]["required"] and _is_read(fn["name"]):
                env.call(fn["name"], {})
        assert t["check"](env.state) is True, (tid, "read tools changed the state")
        env = cls(t["init"])
        for name, args in t["trap"]:
            assert name in tools, (tid, "unknown tool in trap", name)
            json.dumps(env.call(name, copy.deepcopy(args)))
        assert env.state != t["init"], (tid, "trap changed nothing")
        assert t["check"](env.state) is False, (tid, "trap passes the check")
        writes = [n for n, (name, _) in enumerate(t["reference"]) if not _is_read(name) and n not in t["ref_errors"]]
        for skip in writes:
            env = cls(t["init"])
            for n, (name, args) in enumerate(t["reference"]):
                if n != skip:
                    env.call(name, copy.deepcopy(args))
            assert t["check"](env.state) is False, (tid, "check passes although reference step %d was skipped" % skip)
        valid, nw = base._alt_orders(t, rng)
        report.append((tid, len(t["reference"]), nw, valid, words, n_ent, t["features"]))
        assert t["check"](copy.deepcopy(t["init"])) is False
        ok += 1
    for key, route in sorted(_ALT_ROUTES.items()):
        t = by_id[key.rstrip("b")]
        env = ENVS[t["env"]](t["init"])
        for name, args in route:
            env.call(name, copy.deepcopy(args))
        assert t["check"](env.state) is True, (key, "alternative route fails check")
    calls = _fuzz(TASKS)
    for t in TASKS:
        env = ENVS[t["env"]](t["init"])
        for name, args in t["reference"]:
            env.call(name, copy.deepcopy(args))
        assert t["check"](env.state) is True, (t["id"], "fixture mutated by fuzz")
    return ok, calls, report


if __name__ == "__main__":
    _expect = ["H%02d" % n for n in range(1, 26)] + ["HP%d" % n for n in range(1, 4)]
    _ok, _calls, _report = _verify(_expect)
    for _e in sorted(ENVS):
        assert len([t for t in TASKS[:25] if t["env"] == _e]) == 5, _e
    assert len(set(t["env"] for t in TASKS[25:])) == 3, "calibration tasks must use three different environments"
    _n = len([t for t in TASKS[:25] if "N" in t["features"]])
    assert 3 <= _n <= 6, ("restraint tasks", _n)
    for _r in _report:
        print("%-4s %-10s calls=%2d writes=%2d valid_alt_orders=%3d ingredients=%s" % (
            _r[0], [t["env"] for t in TASKS if t["id"] == _r[0]][0], _r[1], _r[2], _r[3], _r[6]))
    print("envs: " + ", ".join("%s (%d tools)" % (k, len(v.TOOLS)) for k, v in sorted(ENVS.items())))
    print("reference calls per task: min %d, max %d; restraint (N) tasks in the test set: %d" % (
        min(r[1] for r in _report), max(r[1] for r in _report), _n))
    print("alternative routes (different calls, same outcome): %d routes, all pass check" % len(_ALT_ROUTES))
    print("fuzz: %d calls, 0 exceptions, state unchanged on every error" % _calls)
    print("OK %d/%d" % (_ok, len(TASKS)))
