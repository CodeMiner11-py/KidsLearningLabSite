import { db } from './firebase.js';
import { doc, collection, setDoc } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

// ---- Setup you need to do once, then paste the URL in here ----
// This is the Cloudflare Worker described in notifications-worker/README.md.
// It's a *separate* worker from your email one — deploy it, then replace
// this placeholder with its real *.workers.dev URL (or custom domain).
export const PUSH_WORKER_URL = 'https://kll-notifications-worker.nameless-cherry-998c.workers.dev';

// Writes an in-app notification for `toUid` and (best-effort) asks the
// Cloudflare Worker to also fire an actual push through FCM. The Firestore
// write always happens — even if the push relay fails (offline, worker
// down, user hasn't granted push permission yet, etc.) the person still
// sees it next time they open the bell panel.
//
// type: 'friend_request' | 'friend_accepted' | 'course_shared'
//       (streak reminders / new episode are sent server-side by the worker's
//       cron job, not from the client)
export async function notifyUser(toUid, { type, title, body, data = {} }) {
  if (!toUid || !title) return;
  try {
    await setDoc(doc(collection(db, 'users', toUid, 'notifications')), {
      type, title, body: body || '', data, createdAt: Date.now(), read: false,
    });
  } catch (err) {
    console.error('Failed to write notification:', err);
  }

  // Fire-and-forget: never block the friend-request/share action on this.
  fetch(PUSH_WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uid: toUid, title, body: body || '', data }),
  }).catch((err) => console.warn('Push relay failed (in-app notification still saved):', err));
}