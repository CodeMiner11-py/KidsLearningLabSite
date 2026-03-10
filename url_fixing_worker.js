let path = window.location.pathname;

let corrected = path.replace(/\.+$/, "");

if (corrected !== path) {
    fetch(corrected, { method: "HEAD" })
        .then(response => {
            if (response.ok) {
                window.location.replace(corrected + window.location.search + window.location.hash);
            }
        })
        .catch(() => {});
}