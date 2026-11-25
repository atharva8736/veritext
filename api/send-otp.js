// Vercel serverless function: emails a 6-digit sign-in code through Brevo.
// Needs these environment variables in the Vercel project:
//   BREVO_API_KEY       Brevo API key (SMTP & API > API Keys)
//   BREVO_SENDER_EMAIL  a sender address verified in Brevo
//   OTP_SECRET          any long random string (signs the code)
const crypto = require('crypto');

module.exports = async (req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });
  const key = process.env.BREVO_API_KEY, sender = process.env.BREVO_SENDER_EMAIL, secret = process.env.OTP_SECRET;
  if (!key || !sender || !secret) return res.status(500).json({ error: 'Email service is not configured yet.' });

  const email = String((req.body && req.body.email) || '').trim().toLowerCase();
  if (email.length > 254 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return res.status(400).json({ error: 'Enter a valid email address.' });
  }

  const code = String(crypto.randomInt(0, 1000000)).padStart(6, '0');
  const exp = Date.now() + 10 * 60 * 1000;
  const sig = crypto.createHmac('sha256', secret).update(`${email}|${exp}|${code}`).digest('hex');
  const token = Buffer.from(JSON.stringify({ e: exp, s: sig })).toString('base64url');

  const html = `<div style="font-family:Arial,sans-serif;max-width:420px;margin:auto;padding:24px;color:#17202B">
    <h2 style="color:#14213D;margin:0 0 12px">Veritext</h2>
    <p>Your sign-in code is:</p>
    <p style="font-size:34px;letter-spacing:8px;font-weight:700;color:#14213D;margin:12px 0">${code}</p>
    <p style="color:#566070">It expires in 10 minutes. If you did not ask for it, you can ignore this email.</p></div>`;
  try {
    const r = await fetch('https://api.brevo.com/v3/smtp/email', {
      method: 'POST',
      headers: { 'api-key': key, 'content-type': 'application/json', accept: 'application/json' },
      body: JSON.stringify({
        sender: { name: 'Veritext', email: sender },
        to: [{ email }],
        subject: 'Your Veritext sign-in code',
        htmlContent: html,
        textContent: `Your Veritext sign-in code is ${code}. It expires in 10 minutes.`
      })
    });
    if (!r.ok) return res.status(502).json({ error: 'The email could not be sent. Please try again.' });
  } catch (e) {
    return res.status(502).json({ error: 'The email service is unreachable. Please try again.' });
  }
  return res.status(200).json({ token });
};
