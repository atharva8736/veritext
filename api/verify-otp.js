// Vercel serverless function: checks the 6-digit code against the signed token.
const crypto = require('crypto');

module.exports = (req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });
  const secret = process.env.OTP_SECRET;
  if (!secret) return res.status(500).json({ error: 'Email service is not configured yet.' });

  const b = req.body || {};
  const email = String(b.email || '').trim().toLowerCase();
  const code = String(b.code || '').trim();
  if (!/^\d{6}$/.test(code)) return res.status(400).json({ error: 'Enter the 6-digit code.' });

  let t;
  try { t = JSON.parse(Buffer.from(String(b.token || ''), 'base64url').toString()); } catch (e) { t = null; }
  if (!t || typeof t.e !== 'number' || typeof t.s !== 'string') return res.status(400).json({ error: 'Please request a new code.' });
  if (Date.now() > t.e) return res.status(400).json({ error: 'That code has expired. Request a new one.' });

  const want = crypto.createHmac('sha256', secret).update(`${email}|${t.e}|${code}`).digest();
  let got;
  try { got = Buffer.from(t.s, 'hex'); } catch (e) { got = Buffer.alloc(0); }
  if (got.length !== want.length || !crypto.timingSafeEqual(got, want)) return res.status(401).json({ error: 'Wrong code. Please check and try again.' });
  return res.status(200).json({ ok: true });
};
