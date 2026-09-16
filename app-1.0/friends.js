// friends.js — Kids Learning Lab friend-request data layer (no DOM)
//
// The actual Firestore writes behind "send a friend request", pulled out
// of main.js so learn.js can reuse them too: when a QR scanned from the
// "Join a Game" or "Share Course" screens turns out to be a person's code
// rather than a game/course code (see qrRouting.js), the right move is to
// send a friend request instead of failing — and that needs to work from
// learn.js without learn.js importing main.js, which would be circular
// (main.js already imports learn.js).
//
// main.js still owns everything DOM-related (the friend request list, its
// buttons, the live-refresh after a change) — this module only ever
// touches Firestore.
import { auth, db } from './firebase.js';
import { doc, getDoc, setDoc } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";
import { notifyUser } from './notifications.js';

// Sends (or, if they'd already requested us, auto-accepts) a friend
// request to `otherUid`. Identical underlying writes no matter how
// otherUid was discovered — typed email lookup, a scanned "My QR Code",
// or a QR scanned somewhere else that turned out to be a person's code.
//
// opts.fromDisplayName lets a caller that already has display-name/premium
// formatting (main.js) pass its exact formatted name through for the
// notification body; callers without that context get a plain fallback.
//
// Returns { outcome: 'accepted' | 'sent' } on success.
// Throws Error('...') with a user-facing message on failure.
export async function sendFriendRequestToUid(otherUid, opts = {}) {
  const u = auth.currentUser;
  if (!u) throw new Error('Not signed in.');
  if (otherUid === u.uid) throw new Error("That's you!");

  const fromDisplayName = opts.fromDisplayName
    || u.displayName
    || (u.email ? u.email.split('@')[0] : 'Someone');

  // If they already sent *us* a request, accept it instead of creating a
  // duplicate reverse request.
  const existingReverse = await getDoc(doc(db, 'users', u.uid, 'friends', otherUid));
  if (existingReverse.exists() && existingReverse.data().status === 'pending' && existingReverse.data().direction === 'received') {
    await setDoc(doc(db, 'users', u.uid, 'friends', otherUid), { status: 'accepted' }, { merge: true });
    await setDoc(doc(db, 'users', otherUid, 'friends', u.uid), { status: 'accepted' }, { merge: true });
    notifyUser(otherUid, {
      type: 'friend_accepted',
      title: 'Friend request accepted',
      body: `${fromDisplayName} accepted your friend request`,
      data: { fromUid: u.uid },
    });
    return { outcome: 'accepted' };
  }
  if (existingReverse.exists() && existingReverse.data().status === 'accepted') {
    throw new Error("You're already friends!");
  }

  await setDoc(doc(db, 'users', u.uid, 'friends', otherUid), {
    status: 'pending', direction: 'sent', createdAt: Date.now(),
  });
  await setDoc(doc(db, 'users', otherUid, 'friends', u.uid), {
    status: 'pending', direction: 'received', createdAt: Date.now(),
  });
  notifyUser(otherUid, {
    type: 'friend_request',
    title: 'New friend request',
    body: `${fromDisplayName} wants to be friends`,
    data: { fromUid: u.uid },
  });
  return { outcome: 'sent' };
}