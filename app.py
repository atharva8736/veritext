"""Veritext website. Run:  python app.py   then open http://127.0.0.1:5000"""
import os, json, sqlite3, secrets
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, g, flash, abort
from werkzeug.security import generate_password_hash, check_password_hash
from detector_service import Service, verdict, read_upload

app = Flask(__name__, instance_relative_config=True)
os.makedirs(app.instance_path, exist_ok=True)
key_file = os.path.join(app.instance_path, "secret.key")
if not os.path.exists(key_file): open(key_file, "w").write(secrets.token_hex(32))
app.secret_key = open(key_file).read()
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
DB = os.path.join(app.instance_path, "veritext.db")
svc = Service()

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB); g.db.row_factory = sqlite3.Row
    return g.db
@app.teardown_appcontext
def close(_):
    d = g.pop("db", None)
    if d: d.close()

def init_db():
    c = sqlite3.connect(DB)
    c.executescript("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, pw TEXT, mode TEXT DEFAULT 'fusion', thr INTEGER DEFAULT 50);
    CREATE TABLE IF NOT EXISTS checks(id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, source TEXT, text TEXT, p_ai REAL, verdict TEXT, mode TEXT, created TEXT);""")
    if not c.execute("SELECT 1 FROM users WHERE email='athrav@demo.com'").fetchone():
        c.execute("INSERT INTO users(name,email,pw) VALUES(?,?,?)", ("Athrav Singh", "athrav@demo.com", generate_password_hash("demo1234")))
    c.commit(); c.close()
init_db()

def login_required(f):
    @wraps(f)
    def w(*a, **k):
        if "uid" not in session: return redirect(url_for("login"))
        g.user = db().execute("SELECT * FROM users WHERE id=?", (session["uid"],)).fetchone()
        if not g.user: session.clear(); return redirect(url_for("login"))
        return f(*a, **k)
    return w

@app.template_filter("pct")
def pct(p): return f"{round(p*100)}"
@app.template_filter("initials")
def initials(n): return "".join(w[0] for w in n.split()[:2]).upper()
@app.template_filter("when")
def when(s):
    d = datetime.fromisoformat(s); days = (datetime.now().date() - d.date()).days
    return "Today" if days == 0 else "Yesterday" if days == 1 else f"{days} days ago" if days < 7 else d.strftime("%d %b")

@app.route("/")
def splash():
    if "uid" in session: return redirect(url_for("dashboard"))
    return render_template("splash.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = db().execute("SELECT * FROM users WHERE email=?", (request.form["email"].strip().lower(),)).fetchone()
        if u and check_password_hash(u["pw"], request.form["password"]):
            session["uid"] = u["id"]; return redirect(url_for("dashboard"))
        flash("Wrong email or password.")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name, email, pw = request.form["name"].strip(), request.form["email"].strip().lower(), request.form["password"]
        if not name or "@" not in email or len(pw) < 6:
            flash("Enter your name, a valid email and a password of at least 6 characters.")
        else:
            try:
                cur = db().execute("INSERT INTO users(name,email,pw) VALUES(?,?,?)", (name, email, generate_password_hash(pw)))
                db().commit(); session["uid"] = cur.lastrowid; return redirect(url_for("dashboard"))
            except sqlite3.IntegrityError:
                flash("That email is already registered.")
    return render_template("register.html")

@app.route("/logout")
def logout(): session.clear(); return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    rows = db().execute("SELECT * FROM checks WHERE user_id=? ORDER BY id DESC", (g.user["id"],)).fetchall()
    stats = {"total": len(rows), "ai": sum(r["verdict"] == "ai" for r in rows),
             "conf": round(100 * sum(max(r["p_ai"], 1 - r["p_ai"]) for r in rows) / len(rows)) if rows else 0}
    return render_template("dashboard.html", active="home", rows=rows[:4], stats=stats)

@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    if request.method == "POST":
        text, source, title = request.form.get("text", "").strip(), "Pasted", "Pasted text"
        f = request.files.get("file")
        if f and f.filename:
            text, source, title = read_upload(f), "Uploaded", f.filename
        if len(text.split()) < 8:
            flash("Please enter at least 8 words so the result is meaningful."); return render_template("analyze.html", active="scan", text=text, mode=request.form.get("mode", g.user["mode"]))
        if source == "Pasted": title = " ".join(text.split()[:5]) + "..."
        mode = request.form.get("mode", g.user["mode"])
        p = svc.p_ai(text, mode)
        cur = db().execute("INSERT INTO checks(user_id,title,source,text,p_ai,verdict,mode,created) VALUES(?,?,?,?,?,?,?,?)",
                           (g.user["id"], title, source, text, p, verdict(p, g.user["thr"] / 100), mode, datetime.now().isoformat()))
        db().commit(); return redirect(url_for("result", cid=cur.lastrowid))
    return render_template("analyze.html", active="scan", text="", mode=g.user["mode"])

def get_check(cid):
    r = db().execute("SELECT * FROM checks WHERE id=? AND user_id=?", (cid, g.user["id"])).fetchone()
    if not r: abort(404)
    return r, svc.analyze(r["text"], r["mode"])

@app.route("/result/<int:cid>")
@login_required
def result(cid):
    r, a = get_check(cid); return render_template("result.html", active="scan", r=r, a=a)

@app.route("/report/<int:cid>")
@login_required
def report(cid):
    r, a = get_check(cid); return render_template("report.html", active="scan", r=r, a=a)

@app.route("/history")
@login_required
def history():
    q, v = request.args.get("q", "").strip().lower(), request.args.get("v", "all")
    rows = [r for r in db().execute("SELECT * FROM checks WHERE user_id=? ORDER BY id DESC", (g.user["id"],)).fetchall()
            if (not q or q in r["title"].lower()) and (v == "all" or r["verdict"] == v)]
    return render_template("history.html", active="history", rows=rows, q=q, v=v)

@app.route("/delete/<int:cid>", methods=["POST"])
@login_required
def delete(cid):
    db().execute("DELETE FROM checks WHERE id=? AND user_id=?", (cid, g.user["id"])); db().commit()
    return redirect(url_for("history"))

@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        mode = request.form.get("mode", "fusion"); thr = int(request.form.get("thr", 50))
        db().execute("UPDATE users SET mode=?, thr=? WHERE id=?", (mode if mode in ("stylometric", "transformer", "fusion") else "fusion", min(max(thr, 30), 70), g.user["id"]))
        db().commit(); flash("Settings saved."); return redirect(url_for("settings"))
    return render_template("settings.html", active="settings", backend=svc.backend)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
