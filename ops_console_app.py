# ops_console_app.py
import os
from datetime import datetime
from functools import wraps

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    send_file,
)

import io
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId

from mongodb_handler import MongoDBHandler

# -------------------------------------------------
# Basic config
# -------------------------------------------------
MONGODB_URI = os.environ.get(
    "MONGODB_URI",
    "mongodb+srv://admin:Admin123456@cluster0.z8yhqsg.mongodb.net/cash_recon?retryWrites=true&w=majority&appName=Cluster0",
)
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "cash_recon")

mongo_handler = MongoDBHandler(MONGODB_URI, MONGODB_DB_NAME)

# direct client for ops-specific collection
mongo_client = MongoClient(MONGODB_URI)
ops_db = mongo_client[MONGODB_DB_NAME]
ops_cleared_col = ops_db["ops_cleared_breaks"]
ops_users_col = ops_db["ops_users"]
ops_assigned_col = ops_db["ops_assigned_breaks"]  # NEW



app = Flask(__name__)
app.secret_key = os.environ.get("OPS_CONSOLE_SECRET", "change-me-ops-console")

# hard-coded ops users (you can later move this to Mongo)
USERS = {
    "ops1": "Password123",
    "ops2": "Password123",
}

BROKER_LABELS = [
    "Velocity",
    "Clear Street",
    "SCB",
    "Riyadh Capital",
    "GTNA",
]

# -------------------------------------------------
# Helpers shared with Cash Recon Pro
# -------------------------------------------------

def ensure_default_admin():
    """
    Make sure we have at least one admin user for login.
    username: admin
    password: Admin123
    """
    exists = ops_users_col.find_one({"username": "admin"})
    if not exists:
        ops_users_col.insert_one(
            {
                "username": "admin",
                "password": "Admin123",  # you can change later
                "role": "admin",
                "created_at": datetime.utcnow(),
            }
        )

# Call this once at import time
ensure_default_admin()


def _pick_broker_key(name: str) -> str:
    k = (name or "").strip().lower()
    alias = {
        "velocity": "velocity",
        "clear street": "clearstreet",
        "clearstreet": "clearstreet",
        "scb": "scb",
        "standard chartered": "scb",
        "standard chartered bank": "scb",
        "riyadh capital": "riyadh capital",
        "riyadhcapital": "riyadh capital",
        "rc": "riyadh capital",
        "gtna": "gtna",
        "gtn a": "gtna",
        "gtn": "gtna",
        "gtn asia": "gtna",
    }
    return alias.get(k, "velocity")


def make_rec_key(account: str, broker: str) -> str:
    account = (account or "").strip()
    broker = (broker or "").strip()
    if not account:
        return "default"
    if not broker:
        return account
    return f"{account}__{broker}"


def _build_signature(date_val, symbol, desc, at, brk) -> str:
    """
    Build a stable string key for a row so we can match
    between rec, history, and ops metadata.
    """
    try:
        if pd.isna(date_val):
            d_str = ""
        else:
            d_str = pd.to_datetime(date_val).strftime("%Y-%m-%d")
    except Exception:
        d_str = str(date_val) or ""
    s = (symbol or "").strip()
    d = (desc or "").strip()
    try:
        a = float(at)
    except Exception:
        a = 0.0
    try:
        b = float(brk)
    except Exception:
        b = 0.0
    return f"{d_str}|{s}|{d}|{a:.2f}|{b:.2f}"


def _strip_break_cleared_marker(comment: str) -> str:
    """
    Remove any 'Break Cleared...' marker from a comment.
    Keep the rest of the free text.
    """
    if not isinstance(comment, str):
        return ""
    parts = comment.split("Break Cleared on")
    # keep anything before the marker
    return parts[0].strip()


# -------------------------------------------------
# Auth helpers
# -------------------------------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "ops_user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper
def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "ops_user" not in session:
            return redirect(url_for("login"))
        if session.get("ops_role") != "admin":
            # Non-admins go back to main console
            return redirect(url_for("console"))
        return f(*args, **kwargs)
    return wrapper



# -------------------------------------------------
# Login / Logout
# -------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        user_doc = ops_users_col.find_one({"username": username})

        # VERY simple check: plain password compare
        if user_doc and user_doc.get("password") == password:
            session["ops_user"] = user_doc["username"]
            session["ops_role"] = user_doc.get("role", "user")
            return redirect(url_for("console"))
        else:
            error = "Invalid username or password"

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("ops_user", None)
    return redirect(url_for("login"))

# -------------------------------------------------
# ADMIN: Manage Users (Add / Remove)
# -------------------------------------------------
@app.route("/admin/users", methods=["GET"])
@admin_required
def admin_users():
    users = list(
        ops_users_col.find({}, {"_id": 0, "username": 1, "role": 1})
    )
    return render_template(
        "admin_users.html",
        users=users,
        ops_user=session.get("ops_user"),
    )


@app.route("/admin/users/add", methods=["POST"])
@admin_required
def admin_add_user():
    username = (request.form.get("username") or "").strip()
    password = (request.form.get("password") or "").strip()
    role = (request.form.get("role") or "user").strip().lower()

    if not username or not password:
        return redirect(url_for("admin_users"))

    # prevent duplicate usernames
    exists = ops_users_col.find_one({"username": username})
    if exists:
        return redirect(url_for("admin_users"))

    if role not in ("admin", "user"):
        role = "user"

    ops_users_col.insert_one(
        {
            "username": username,
            "password": password,
            "role": role,
            "created_at": datetime.utcnow(),
        }
    )
    return redirect(url_for("admin_users"))


@app.route("/admin/users/delete", methods=["POST"])
@admin_required
def admin_delete_user():
    username = (request.form.get("username") or "").strip()
    # don't allow deleting yourself or the last admin in a real setup,
    # but for now keep it simple
    if username:
        ops_users_col.delete_one({"username": username})
    return redirect(url_for("admin_users"))



# -------------------------------------------------
# Main console screen
# -------------------------------------------------
@app.route("/")
@login_required
def console():
    accounts = mongo_handler.load_accounts_list()
    return render_template(
        "console.html",
        accounts=accounts,
        brokers=BROKER_LABELS,
        ops_user=session.get("ops_user"),
        ops_role=session.get("ops_role", "user"),
    )


# -------------------------------------------------
# API: Load Outstanding & Cleared
# -------------------------------------------------
@app.route("/api/load_breaks", methods=["POST"])
@login_required
def load_breaks():
    try:
        data = request.get_json(force=True) or {}
        account = (data.get("account") or "").strip()
        broker_label = (data.get("broker") or "").strip()

        if not account or not broker_label:
            return jsonify(ok=False, error="Please select both Account & Broker"), 400

        broker_key = _pick_broker_key(broker_label)
        rec_key = make_rec_key(account, broker_key)

        # 1) Load current reconciliation for this Account+Broker
        df = mongo_handler.load_session_rec(rec_key)
        if df is None or df.empty:
            # No rec built yet or not saved for this combo
            return jsonify(
                ok=True,
                outstanding=[],
                cleared=[],
                message=(
                    "No reconciliation found for this Account + Broker. "
                    "Ask Rec team to build and save the rec first."
                ),
            )

        # Ensure columns exist
        for col in ["OurFlag", "BrkFlag", "Comments", "RowID", "Date",
                    "Symbol", "Description", "AT", "Broker"]:
            if col not in df.columns:
                if col == "RowID":
                    df[col] = range(1, len(df) + 1)
                else:
                    df[col] = "" if col not in ("AT", "Broker") else 0.0

        df["OurFlag"] = df["OurFlag"].fillna("").astype(str)
        df["BrkFlag"] = df["BrkFlag"].fillna("").astype(str)
        df["Comments"] = df["Comments"].fillna("").astype(str)

        # Only consider current UNMATCHED rows as "breaks"
        mask_unmatched = (df["OurFlag"] == "") & (df["BrkFlag"] == "")
        df_breaks = df.loc[mask_unmatched].copy()

        # If no breaks, just return empty sets
        if df_breaks.empty:
            return jsonify(
                ok=True,
                outstanding=[],
                cleared=[],
                message="No outstanding breaks found for this Account + Broker.",
            )

        # Build signature for each break
        df_breaks["Signature"] = df_breaks.apply(
            lambda r: _build_signature(
                r.get("Date"),
                r.get("Symbol"),
                r.get("Description"),
                r.get("AT"),
                r.get("Broker"),
            ),
            axis=1,
        )

        # 2) Load ops metadata for cleared breaks (for this account+broker)
        ops_docs = list(
            ops_cleared_col.find(
                {"account": account, "broker": broker_key},
                {"_id": 0, "signature": 1, "cleared_by": 1, "cleared_at": 1},
            )
        )
        ops_by_sig = {d["signature"]: d for d in ops_docs}

        # ---------- NEW LOGIC FOR OUTSTANDING ----------

        # Outstanding tab:
        # ALL current breaks, with:
        # - status = "OPEN"               (no ops_cleared record)
        # - status = "AWAITING_REC_MATCH" (ops cleared but rec not matched yet)
        today = datetime.utcnow().date()
        outstanding_rows = []

        for _, r in df_breaks.iterrows():
            sig = r["Signature"]
            has_ops_clear = sig in ops_by_sig
            status = "AWAITING_REC_MATCH" if has_ops_clear else "OPEN"

            # Parse Date and compute age
            date_val = pd.to_datetime(r.get("Date"), errors="coerce")
            if pd.notna(date_val):
                date_str = date_val.strftime("%Y-%m-%d")
                age_days = (today - date_val.date()).days
            else:
                date_str = ""
                age_days = ""

            outstanding_rows.append(
                {
                    "rowid": int(r.get("RowID")),
                    "signature": sig,
                    "date": date_str,
                    "age": age_days,
                    "symbol": str(r.get("Symbol") or ""),
                    "description": str(r.get("Description") or ""),
                    "at": float(r.get("AT") or 0.0),
                    "broker": float(r.get("Broker") or 0.0),
                    "comments": str(r.get("Comments") or ""),
                    "status": status,
                }
            )

        # ---------- CLEARED TAB: ONLY MATCHED IN REC HISTORY ----------

        hist_df = mongo_handler.load_history(account)
        if hist_df is None:
            hist_df = pd.DataFrame()

        cleared_matched_rows = []
        if not hist_df.empty:
            for col in [
                "Date",
                "Symbol",
                "Description",
                "AT",
                "Broker",
                "Comments",
                "MatchID",
                "SavedAt",
            ]:
                if col not in hist_df.columns:
                    hist_df[col] = ""

            # Build signatures for history rows
            hist_df["Signature"] = hist_df.apply(
                lambda r: _build_signature(
                    r.get("Date"),
                    r.get("Symbol"),
                    r.get("Description"),
                    r.get("AT"),
                    r.get("Broker"),
                ),
                axis=1,
            )

            # Only those that were cleared by Ops for this account+broker
            hist_cleared = hist_df.loc[
                hist_df["Signature"].isin(ops_by_sig.keys())
            ].copy()

            for _, r in hist_cleared.iterrows():
                sig = r["Signature"]
                meta = ops_by_sig.get(sig, {})

                cleared_matched_rows.append(
                    {
                        "source": "history",
                        "rowid": None,
                        "signature": sig,
                        "date": str(r.get("Date") or ""),
                        "symbol": str(r.get("Symbol") or ""),
                        "description": str(r.get("Description") or ""),
                        "at": float(r.get("AT") or 0.0),
                        "broker": float(r.get("Broker") or 0.0),
                        "comments": str(r.get("Comments") or ""),
                        "cleared_by": meta.get("cleared_by", ""),
                        "cleared_at": meta.get("cleared_at").strftime("%Y-%m-%d %H:%M")
                        if meta.get("cleared_at")
                        else "",
                        "matched_date": str(r.get("SavedAt") or ""),
                        "status": "MATCHED_IN_REC",
                    }
                )

        # Cleared tab: ONLY rows that are matched in Rec history
        cleared_rows = cleared_matched_rows

        return jsonify(ok=True, outstanding=outstanding_rows, cleared=cleared_rows)

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500


# -------------------------------------------------
# API: Mark selected rows as CLEARED
# -------------------------------------------------
@app.route("/api/clear_rows", methods=["POST"])
@login_required
def clear_rows():
    try:
        data = request.get_json(force=True) or {}
        account = (data.get("account") or "").strip()
        broker_label = (data.get("broker") or "").strip()
        rows = data.get("rows") or []

        if not account or not broker_label:
            return jsonify(ok=False, error="Account and Broker are required"), 400
        if not rows:
            return jsonify(ok=False, error="No rows selected"), 400

        broker_key = _pick_broker_key(broker_label)
        rec_key = make_rec_key(account, broker_key)

        df = mongo_handler.load_session_rec(rec_key)
        if df is None or df.empty:
            return jsonify(ok=False, error="No reconciliation found"), 400

        for col in ["RowID", "Comments", "Date", "Symbol", "Description", "AT", "Broker"]:
            if col not in df.columns:
                if col == "RowID":
                    df[col] = range(1, len(df) + 1)
                else:
                    df[col] = "" if col not in ("AT", "Broker") else 0.0

        df["Comments"] = df["Comments"].fillna("").astype(str)

        rowid_to_idx = dict(zip(df["RowID"].astype(int), df.index))

        ops_user = session.get("ops_user", "OPS")

        now = datetime.utcnow()
        for row in rows:
            rowid = int(row.get("rowid"))
            idx = rowid_to_idx.get(rowid)
            if idx is None:
                continue

            # Update comment in Rec
            existing = df.at[idx, "Comments"] or ""
            # Avoid duplicating marker
            if "Break Cleared on" not in existing:
                marker = f"Break Cleared on {now.strftime('%Y-%m-%d')} - OPS"
                if existing.strip():
                    new_comment = existing.strip() + " | " + marker
                else:
                    new_comment = marker
                df.at[idx, "Comments"] = new_comment

            # Insert/Update ops metadata
            sig = _build_signature(
                df.at[idx, "Date"],
                df.at[idx, "Symbol"],
                df.at[idx, "Description"],
                df.at[idx, "AT"],
                df.at[idx, "Broker"],
            )
            ops_cleared_col.update_one(
                {"account": account, "broker": broker_key, "signature": sig},
                {
                    "$set": {
                        "account": account,
                        "broker": broker_key,
                        "signature": sig,
                        "cleared_by": ops_user,
                        "cleared_at": now,
                        "updated_at": now,
                    }
                },
                upsert=True,
            )

        mongo_handler.save_session_rec(
            rec_key, df, metadata={"updated_by_ops": ops_user})

        return jsonify(ok=True)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500


# -------------------------------------------------
# API: Undo (remove cleared marker + metadata)
# -------------------------------------------------
@app.route("/api/undo_rows", methods=["POST"])
@login_required
def undo_rows():
    try:
        data = request.get_json(force=True) or {}
        account = (data.get("account") or "").strip()
        broker_label = (data.get("broker") or "").strip()
        rows = data.get("rows") or []

        if not account or not broker_label:
            return jsonify(ok=False, error="Account and Broker are required"), 400
        if not rows:
            return jsonify(ok=False, error="No rows selected"), 400

        broker_key = _pick_broker_key(broker_label)
        rec_key = make_rec_key(account, broker_key)

        # Load rec (for rows coming from rec)
        df = mongo_handler.load_session_rec(rec_key)
        if df is None:
            df = pd.DataFrame()

        for col in ["RowID", "Comments", "Date", "Symbol", "Description", "AT", "Broker"]:
            if col not in df.columns:
                if col == "RowID":
                    df[col] = range(1, len(df) + 1) if not df.empty else []
                else:
                    df[col] = "" if col not in ("AT", "Broker") else 0.0

        df["Comments"] = df["Comments"].fillna("").astype(str)
        rowid_to_idx = dict(zip(df["RowID"].astype(int), df.index))

        # Load history (for rows coming from history)
        hist_df = mongo_handler.load_history(account)
        if hist_df is None:
            hist_df = pd.DataFrame()

        for col in ["Date", "Symbol", "Description", "AT", "Broker", "Comments"]:
            if col not in hist_df.columns:
                hist_df[col] = ""
        hist_df["Comments"] = hist_df["Comments"].fillna("").astype(str)

        for row in rows:
            sig = row.get("signature")
            source = row.get("source")

            # 1) remove metadata
            ops_cleared_col.delete_one(
                {"account": account, "broker": broker_key, "signature": sig}
            )

            # 2) update comments in Rec or History
            if source == "rec":
                rowid = row.get("rowid")
                if rowid is not None:
                    idx = rowid_to_idx.get(int(rowid))
                    if idx is not None:
                        current = df.at[idx, "Comments"]
                        df.at[idx, "Comments"] = _strip_break_cleared_marker(
                            current)
            elif source == "history":
                # update all matching rows with this signature
                mask = hist_df.apply(
                    lambda r: _build_signature(
                        r.get("Date"), r.get("Symbol"), r.get(
                            "Description"), r.get("AT"), r.get("Broker")
                    )
                    == sig,
                    axis=1,
                )
                hist_df.loc[mask, "Comments"] = hist_df.loc[mask, "Comments"].apply(
                    _strip_break_cleared_marker
                )

        # Save back
        if not df.empty:
            mongo_handler.save_session_rec(
                rec_key, df, metadata={"ops_undo": True})
        if not hist_df.empty:
            mongo_handler.save_history(account, hist_df)

        return jsonify(ok=True)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500
    
    # -------------------------------------------------
# ADMIN: Aged Breaks (20+ days)
# -------------------------------------------------
@app.route("/admin/aged_breaks", methods=["GET"])
@admin_required
def admin_aged_breaks():
    min_age = 20
    today = datetime.utcnow().date()

    # optional filters from query string
    account_filter = (request.args.get("account") or "").strip()
    broker_filter = (request.args.get("broker") or "").strip()

    aged_rows = []

    # accounts list for dropdown
    accounts = mongo_handler.load_accounts_list() or []

    # Pre-load assigned breaks so we can show "Assigned to X"
    assigned_docs = list(
        ops_assigned_col.find(
            {"status": "OPEN"},
            {"_id": 0, "account": 1, "broker": 1, "signature": 1, "assigned_to": 1},
        )
    )
    assigned_lookup = {
        (d["account"], d["broker"], d["signature"]): d["assigned_to"]
        for d in assigned_docs
    }

    for account in accounts:
        # filter by account if chosen
        if account_filter and account != account_filter:
            continue

        for broker_label in BROKER_LABELS:
            if broker_filter and broker_label != broker_filter:
                continue

            broker_key = _pick_broker_key(broker_label)
            rec_key = make_rec_key(account, broker_key)
            df = mongo_handler.load_session_rec(rec_key)
            if df is None or df.empty:
                continue

            for col in [
                "OurFlag",
                "BrkFlag",
                "Comments",
                "RowID",
                "Date",
                "Symbol",
                "Description",
                "AT",
                "Broker",
            ]:
                if col not in df.columns:
                    if col == "RowID":
                        df[col] = range(1, len(df) + 1)
                    else:
                        df[col] = "" if col not in ("AT", "Broker") else 0.0

            df["OurFlag"] = df["OurFlag"].fillna("").astype(str)
            df["BrkFlag"] = df["BrkFlag"].fillna("").astype(str)
            df["Comments"] = df["Comments"].fillna("").astype(str)

            mask_unmatched = (df["OurFlag"] == "") & (df["BrkFlag"] == "")
            df_breaks = df.loc[mask_unmatched].copy()
            if df_breaks.empty:
                continue

            # build signatures
            df_breaks["Signature"] = df_breaks.apply(
                lambda r: _build_signature(
                    r.get("Date"),
                    r.get("Symbol"),
                    r.get("Description"),
                    r.get("AT"),
                    r.get("Broker"),
                ),
                axis=1,
            )

            for _, r in df_breaks.iterrows():
                date_val = pd.to_datetime(r.get("Date"), errors="coerce")
                if pd.notna(date_val):
                    age_days = (today - date_val.date()).days
                    date_str = date_val.strftime("%Y-%m-%d")
                else:
                    age_days = None
                    date_str = ""

                if age_days is None or age_days < min_age:
                    continue

                sig = r["Signature"]
                assigned_to = assigned_lookup.get((account, broker_key, sig), "")

                aged_rows.append(
                    {
                        "account": account,
                        "broker_label": broker_label,
                        "broker_key": broker_key,
                        "date": date_str,
                        "age": age_days,
                        "symbol": str(r.get("Symbol") or ""),
                        "description": str(r.get("Description") or ""),
                        "at": float(r.get("AT") or 0.0),
                        "broker_amt": float(r.get("Broker") or 0.0),
                        "comments": str(r.get("Comments") or ""),
                        "signature": sig,
                        "assigned_to": assigned_to,
                    }
                )

    aged_rows.sort(key=lambda x: x["age"], reverse=True)

    # load all users for Assign dropdown
    users = list(
        ops_users_col.find({}, {"_id": 0, "username": 1, "role": 1})
    )

    return render_template(
        "admin_aged_breaks.html",
        rows=aged_rows,
        ops_user=session.get("ops_user"),
        ops_role=session.get("ops_role", "user"),
        min_age=min_age,
        accounts=accounts,
        brokers=BROKER_LABELS,
        users=users,
        account_filter=account_filter,
        broker_filter=broker_filter,
    )

@app.route("/admin/assign_breaks", methods=["POST"])
@admin_required
def admin_assign_breaks():
    assignee = (request.form.get("assignee") or "").strip()
    selected = request.form.getlist("rows")  # each value: account||broker_key||signature

    if not assignee or not selected:
        # just go back – in real app you would flash a message
        return redirect(url_for("admin_aged_breaks"))

    admin_user = session.get("ops_user", "admin")
    now = datetime.utcnow()

    for item in selected:
        try:
            account, broker_key, signature = item.split("||", 2)
        except ValueError:
            continue

        ops_assigned_col.update_one(
            {
                "account": account,
                "broker": broker_key,
                "signature": signature,
            },
            {
                "$set": {
                    "account": account,
                    "broker": broker_key,
                    "signature": signature,
                    "assigned_to": assignee,
                    "assigned_by": admin_user,
                    "assigned_at": now,
                    "status": "OPEN",
                }
            },
            upsert=True,
        )

    return redirect(url_for("admin_aged_breaks"))

@app.route("/assigned", methods=["GET"])
@login_required
def my_assigned_breaks():
    accounts = mongo_handler.load_accounts_list()
    return render_template(
        "my_assigned_breaks.html",
        accounts=accounts,
        brokers=BROKER_LABELS,
        ops_user=session.get("ops_user"),
        ops_role=session.get("ops_role", "user"),
    )

@app.route("/api/my_assigned_breaks", methods=["POST"])
@login_required
def api_my_assigned_breaks():
    data = request.get_json(force=True) or {}
    account = (data.get("account") or "").strip()
    broker_label = (data.get("broker") or "").strip()

    if not account or not broker_label:
        return jsonify(ok=False, error="Account and Broker are required"), 400

    broker_key = _pick_broker_key(broker_label)
    rec_key = make_rec_key(account, broker_key)
    username = session.get("ops_user")

    # open assignments for this user / rec
    assigned_docs = list(
        ops_assigned_col.find(
            {
                "account": account,
                "broker": broker_key,
                "assigned_to": username,
                "status": "OPEN",
            },
            {"_id": 0, "signature": 1},
        )
    )
    sigs = {d["signature"] for d in assigned_docs}
    if not sigs:
        return jsonify(ok=True, rows=[])

    df = mongo_handler.load_session_rec(rec_key)
    if df is None or df.empty:
        return jsonify(ok=True, rows=[])

    for col in [
        "OurFlag",
        "BrkFlag",
        "Comments",
        "RowID",
        "Date",
        "Symbol",
        "Description",
        "AT",
        "Broker",
    ]:
        if col not in df.columns:
            if col == "RowID":
                df[col] = range(1, len(df) + 1)
            else:
                df[col] = "" if col not in ("AT", "Broker") else 0.0

    df["OurFlag"] = df["OurFlag"].fillna("").astype(str)
    df["BrkFlag"] = df["BrkFlag"].fillna("").astype(str)
    df["Comments"] = df["Comments"].fillna("").astype(str)

    # only still unmatched
    mask_unmatched = (df["OurFlag"] == "") & (df["BrkFlag"] == "")
    df_breaks = df.loc[mask_unmatched].copy()
    if df_breaks.empty:
        return jsonify(ok=True, rows=[])

    df_breaks["Signature"] = df_breaks.apply(
        lambda r: _build_signature(
            r.get("Date"),
            r.get("Symbol"),
            r.get("Description"),
            r.get("AT"),
            r.get("Broker"),
        ),
        axis=1,
    )

    today = datetime.utcnow().date()
    rows = []
    for _, r in df_breaks.iterrows():
        sig = r["Signature"]
        if sig not in sigs:
            continue

        date_val = pd.to_datetime(r.get("Date"), errors="coerce")
        if pd.notna(date_val):
            date_str = date_val.strftime("%Y-%m-%d")
            age_days = (today - date_val.date()).days
        else:
            date_str = ""
            age_days = ""

        rows.append(
            {
                "rowid": int(r.get("RowID")),
                "signature": sig,
                "date": date_str,
                "age": age_days,
                "symbol": str(r.get("Symbol") or ""),
                "description": str(r.get("Description") or ""),
                "at": float(r.get("AT") or 0.0),
                "broker": float(r.get("Broker") or 0.0),
                "comments": str(r.get("Comments") or ""),
            }
        )

    return jsonify(ok=True, rows=rows)


# -------------------------------------------------
# ADMIN: Cleared Breaks by User (summary per day)
# -------------------------------------------------
@app.route("/admin/cleared_summary", methods=["GET"])
@admin_required
def admin_cleared_summary():
    # date as YYYY-MM-DD, default today
    date_str = request.args.get("date")
    if not date_str:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        day = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        day = datetime.utcnow().date()
        date_str = day.strftime("%Y-%m-%d")

    start_dt = datetime.combine(day, datetime.min.time())
    end_dt = datetime.combine(day, datetime.max.time())

    # get all clears that day
    docs = list(
        ops_cleared_col.find(
            {"cleared_at": {"$gte": start_dt, "$lte": end_dt}},
            {"_id": 0, "cleared_by": 1},
        )
    )

    counts_by_user = {}
    for d in docs:
        u = d.get("cleared_by", "UNKNOWN")
        counts_by_user[u] = counts_by_user.get(u, 0) + 1

    # turn into list for template
    rows = [
        {"username": u, "count": c} for u, c in sorted(
            counts_by_user.items(), key=lambda x: x[0].lower()
        )
    ]

    return render_template(
        "admin_cleared_summary.html",
        rows=rows,
        date_str=date_str,
        ops_user=session.get("ops_user"),
    )

# -------------------------------------------------
# ADMIN: Cleared Breaks detail for one user+day
# -------------------------------------------------
@app.route("/admin/cleared_detail/<username>", methods=["GET"])
@admin_required
def admin_cleared_detail(username):
    date_str = request.args.get("date")
    if not date_str:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        day = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        day = datetime.utcnow().date()
        date_str = day.strftime("%Y-%m-%d")

    start_dt = datetime.combine(day, datetime.min.time())
    end_dt = datetime.combine(day, datetime.max.time())

    # all clears for this user on that day
    docs = list(
        ops_cleared_col.find(
            {
                "cleared_by": username,
                "cleared_at": {"$gte": start_dt, "$lte": end_dt},
            },
            {"_id": 0, "account": 1, "broker": 1},
        )
    )

    counts = {}
    for d in docs:
        acc = d.get("account", "UNKNOWN")
        brk = d.get("broker", "UNKNOWN")
        key = (acc, brk)
        counts[key] = counts.get(key, 0) + 1

    rows = [
        {"account": acc, "broker": brk, "count": c}
        for (acc, brk), c in sorted(counts.items())
    ]

    total = sum(r["count"] for r in rows)

    return render_template(
        "admin_cleared_detail.html",
        username=username,
        date_str=date_str,
        rows=rows,
        total=total,
        ops_user=session.get("ops_user"),
    )



# -------------------------------------------------
# API: Update comment (per-row) – always ends with " - OPS"
# -------------------------------------------------
@app.route("/api/update_comment", methods=["POST"])
@login_required
def api_update_comment():
    try:
        data = request.get_json(force=True) or {}
        account = (data.get("account") or "").strip()
        broker_label = (data.get("broker") or "").strip()
        rowid = data.get("rowid")
        comment = (data.get("comment") or "").strip()

        if not account or not broker_label or rowid is None:
            return jsonify(ok=False, error="Account, Broker and RowID are required"), 400

        broker_key = _pick_broker_key(broker_label)
        rec_key = make_rec_key(account, broker_key)

        df = mongo_handler.load_session_rec(rec_key)
        if df is None or df.empty:
            return jsonify(ok=False, error="No reconciliation found"), 400

        for col in ["RowID", "Comments"]:
            if col not in df.columns:
                if col == "RowID":
                    df[col] = range(1, len(df) + 1)
                else:
                    df[col] = ""

        df["Comments"] = df["Comments"].fillna("").astype(str)
        rowid_to_idx = dict(zip(df["RowID"].astype(int), df.index))

        idx = rowid_to_idx.get(int(rowid))
        if idx is None:
            return jsonify(ok=False, error="RowID not found"), 400

        # Append " - OPS" if not already at the end
        if comment and not comment.endswith("- OPS"):
            if comment.endswith("OPS"):
                final = comment
            else:
                final = comment + " - OPS"
        else:
            final = comment

        df.at[idx, "Comments"] = final
        mongo_handler.save_session_rec(
            rec_key, df, metadata={"comment_updated_by_ops": True})

        return jsonify(ok=True, comment=final)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500


# -------------------------------------------------
# Export Outstanding / Cleared to Excel
# -------------------------------------------------
def _rows_to_df(rows, include_ops_fields=False):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = ["date", "symbol", "description", "at", "broker", "comments"]
    if include_ops_fields:
        cols.extend(["cleared_by", "cleared_at", "matched_date", "status"])
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


@app.route("/export/outstanding", methods=["POST"])
@login_required
def export_outstanding():
    try:
        data = request.get_json(force=True) or {}
        outstanding = data.get("outstanding") or []

        df = _rows_to_df(outstanding, include_ops_fields=False)

        buf = io.BytesIO()

        # Use xlsxwriter so we can control formatting
        with pd.ExcelWriter(
            buf,
            engine="xlsxwriter",
            engine_kwargs={"options": {"nan_inf_to_errors": True}},
        ) as writer:
            wb = writer.book
            ws = wb.add_worksheet("Outstanding")
            writer.sheets["Outstanding"] = ws

            # --------- Formats ----------
            f_header = wb.add_format(
                {
                    "bold": True,
                    "bg_color": "#1f3b57",
                    "font_color": "white",
                    "border": 1,
                    "align": "center",
                    "valign": "vcenter",
                }
            )
            f_text = wb.add_format({"border": 1})
            f_num = wb.add_format(
                {"border": 1, "num_format": "#,##0.00;[Red](#,##0.00)"}
            )

            # If nothing, still create a header-only sheet
            if df is None or df.empty:
                ws.write(0, 0, "No outstanding breaks", f_header)
            else:
                cols = list(df.columns)

                # Header row
                for c_idx, col in enumerate(cols):
                    ws.write(0, c_idx, col.upper(), f_header)

                # Data rows
                for r_idx, (_, row) in enumerate(df.iterrows(), start=1):
                    for c_idx, col in enumerate(cols):
                        val = row.get(col)

                        # numeric formatting for money/amount columns
                        col_lower = col.lower()
                        if col_lower in ("at", "broker", "diff", "difference"):
                            try:
                                num = float(val)
                                ws.write_number(r_idx, c_idx, num, f_num)
                            except Exception:
                                ws.write(
                                    r_idx,
                                    c_idx,
                                    "" if val is None else str(val),
                                    f_text,
                                )
                        else:
                            ws.write(
                                r_idx,
                                c_idx,
                                "" if val is None else str(val),
                                f_text,
                            )

                # Auto-fit-ish column widths
                for c_idx, col in enumerate(cols):
                    max_len = len(str(col))
                    for v in df[col].astype(str):
                        if isinstance(v, str):
                            max_len = max(max_len, len(v[:60]))
                    ws.set_column(c_idx, c_idx, min(max_len + 2, 40))

                # Freeze header row and add filter
                ws.freeze_panes(1, 0)
                ws.autofilter(0, 0, len(df), len(cols) - 1)

        buf.seek(0)
        fname = f"Outstanding_Breaks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return send_file(
            buf,
            as_attachment=True,
            download_name=fname,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return ("Export error: " + str(e), 500)


@app.route("/export/cleared", methods=["POST"])
@login_required
def export_cleared():
    try:
        data = request.get_json(force=True) or {}
        cleared = data.get("cleared") or []

        df = _rows_to_df(cleared, include_ops_fields=True)

        buf = io.BytesIO()

        with pd.ExcelWriter(
            buf,
            engine="xlsxwriter",
            engine_kwargs={"options": {"nan_inf_to_errors": True}},
        ) as writer:
            wb = writer.book
            ws = wb.add_worksheet("Cleared")
            writer.sheets["Cleared"] = ws

            # --------- Formats ----------
            f_header = wb.add_format(
                {
                    "bold": True,
                    "bg_color": "#1f3b57",
                    "font_color": "white",
                    "border": 1,
                    "align": "center",
                    "valign": "vcenter",
                }
            )
            f_text = wb.add_format({"border": 1})
            f_num = wb.add_format(
                {"border": 1, "num_format": "#,##0.00;[Red](#,##0.00)"}
            )

            if df is None or df.empty:
                ws.write(0, 0, "No cleared breaks", f_header)
            else:
                cols = list(df.columns)

                # Header row
                for c_idx, col in enumerate(cols):
                    ws.write(0, c_idx, col.upper(), f_header)

                # Data rows
                for r_idx, (_, row) in enumerate(df.iterrows(), start=1):
                    for c_idx, col in enumerate(cols):
                        val = row.get(col)

                        col_lower = col.lower()
                        if col_lower in ("at", "broker", "diff", "difference"):
                            try:
                                num = float(val)
                                ws.write_number(r_idx, c_idx, num, f_num)
                            except Exception:
                                ws.write(
                                    r_idx,
                                    c_idx,
                                    "" if val is None else str(val),
                                    f_text,
                                )
                        else:
                            ws.write(
                                r_idx,
                                c_idx,
                                "" if val is None else str(val),
                                f_text,
                            )

                # Auto-fit-ish column widths
                for c_idx, col in enumerate(cols):
                    max_len = len(str(col))
                    for v in df[col].astype(str):
                        if isinstance(v, str):
                            max_len = max(max_len, len(v[:60]))
                    ws.set_column(c_idx, c_idx, min(max_len + 2, 40))

                # Freeze header row and add filter
                ws.freeze_panes(1, 0)
                ws.autofilter(0, 0, len(df), len(cols) - 1)

        buf.seek(0)
        fname = f"Cleared_Breaks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return send_file(
            buf,
            as_attachment=True,
            download_name=fname,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return ("Export error: " + str(e), 500)


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))  # 👈 use Railway PORT
    app.run(host="0.0.0.0", port=port, debug=False)


