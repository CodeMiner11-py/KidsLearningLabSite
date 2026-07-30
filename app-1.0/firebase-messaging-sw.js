// firebase-messaging-sw.js
// Must live at the site root (same level as index.html) — the browser only
// lets a service worker control paths at or below where the file itself is
// served from.

importScripts('https://www.gstatic.com/firebasejs/10.12.2/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.12.2/firebase-messaging-compat.js');

// ---- Paste the SAME config object you already have in firebase.js ----
// This has to be duplicated here because service workers can't import your
// app's modules — it's a separate, isolated script context.
firebase.initializeApp({
    apiKey: "AIzaSyAJcn5sgkJbYiQHqfLE45viQ_X32CLUuVI",
    authDomain: "kids-learning-lab-ios-app.firebaseapp.com",
    projectId: "kids-learning-lab-ios-app",
    storageBucket: "kids-learning-lab-ios-app.firebasestorage.app",
    messagingSenderId: "608497010119",
    appId: "1:608497010119:web:c1e1dfced30aa6f51f45e2"
  });

const messaging = firebase.messaging();

// Fires when a push arrives and the app/tab is NOT in the foreground —
// this is what actually shows the OS-level notification.
messaging.onBackgroundMessage((payload) => {
  const title = payload.notification?.title || 'Kids Learning Lab';
  const body = payload.notification?.body || '';
  self.registration.showNotification(title, {
    body,
    icon: './streak.png',
    data: payload.data || {},
  });
});

// Tapping the OS notification focuses/opens the app instead of just
// dismissing it.
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then((all) => {
      const existing = all.find((c) => 'focus' in c);
      if (existing) return existing.focus();
      return clients.openWindow('./');
    })
  );
});