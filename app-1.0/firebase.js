import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import {
  getAuth,
  initializeAuth,
  indexedDBLocalPersistence
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyAJcn5sgkJbYiQHqfLE45viQ_X32CLUuVI",
  authDomain: "kids-learning-lab-ios-app.firebaseapp.com",
  projectId: "kids-learning-lab-ios-app",
  storageBucket: "kids-learning-lab-ios-app.firebasestorage.app",
  messagingSenderId: "608497010119",
  appId: "1:608497010119:web:c1e1dfced30aa6f51f45e2"
};

const app = initializeApp(firebaseConfig);

export const auth = window.Capacitor?.isNativePlatform?.()
  ? initializeAuth(app, { persistence: indexedDBLocalPersistence })
  : getAuth(app);

export const db = getFirestore(app);