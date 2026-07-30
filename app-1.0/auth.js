import { auth } from "./firebase.js";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  sendEmailVerification,
  signOut,
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

const RESET_WORKER_URL = 'https://emailresetpassword-kidslearninglabapp-notpearai.nameless-cherry-998c.workers.dev/'; // TODO: confirm this is the deployed worker URL

// ---- Password reset (used by both "Forgot password?" and the reauth step of Change Password) ----
export async function resetPassword(email) {
  try {
    const res = await fetch(RESET_WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Could not send reset email.');
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message || 'Could not send reset email. Try again.' };
  }
}

// ---- Sign up ----
export async function signUp(email, password) {
  const plugin = window.Capacitor?.Plugins?.FirebaseAuthentication;

  try {
    let result;

    if (plugin?.createUserWithEmailAndPassword) {
      // Native layer creates the account
      await plugin.createUserWithEmailAndPassword({ email, password });

      // Account already exists now — sign in on the web layer instead of creating again
      result = await signInWithEmailAndPassword(auth, email, password);
    } else {
      // No native plugin available — web layer creates the account directly
      result = await createUserWithEmailAndPassword(auth, email, password);
    }

    console.log("Web layer sign-up success:", result?.user?.email);
    await sendEmailVerification(result.user);

    return { success: true, user: result.user };
  } catch (error) {
    return { success: false, error: mapAuthError(error) };
  }
}

// ---- Sign in ----
export async function signIn(email, password) {
  const plugin = window.Capacitor?.Plugins?.FirebaseAuthentication;

  try {
    if (plugin?.signInWithEmailAndPassword) {
      await plugin.signInWithEmailAndPassword({ email, password });
    }

    const result = await signInWithEmailAndPassword(auth, email, password);
    console.log("Web layer sign-in success:", result?.user?.email);

    return { success: true, user: result.user };
  } catch (error) {
    return { success: false, error: mapAuthError(error) };
  }
}

// ---- Sign out ----
export async function logout() {
  const plugin = window.Capacitor?.Plugins?.FirebaseAuthentication;

  if (plugin?.signOut) {
    try {
      await plugin.signOut();
      console.log("Native sign-out successful");
    } catch (e) {
      console.log("Native sign-out error:", e.message);
    }
  }

  await signOut(auth);
}

// ---- Friendly error messages ----
function mapAuthError(error) {
  const code = error?.code || "";
  if (code.includes("email-already-in-use")) return "That email is already registered.";
  if (code.includes("invalid-email")) return "That email address doesn't look right.";
  if (code.includes("weak-password")) return "Password should be at least 6 characters.";
  if (code.includes("user-not-found") || code.includes("wrong-password") || code.includes("invalid-credential")) {
    return "Incorrect email or password.";
  }
  if (code.includes("too-many-requests")) return "Too many attempts. Try again in a bit.";
  return error?.message || "Something went wrong. Please try again.";
}