document.addEventListener("DOMContentLoaded", () => {
    const consoleEl = document.getElementById("consoleLogs");

    // Neon-style log appending
    function appendLog(msg, kind="info") {
        const span = document.createElement("span");
        span.textContent = msg + "\n";
        if(kind === "error") span.style.color = "#ff3860"; // neon red for errors
        else span.style.color = "#00ffe7"; // neon cyan for info
        span.style.textShadow = "0 0 5px currentColor, 0 0 10px currentColor";
        consoleEl.appendChild(span);
        consoleEl.scrollTop = consoleEl.scrollHeight; // auto scroll
    }

    // Placeholder: simulate live logs
    let count = 0;
    const demoLogs = [
        "Initializing BugFiner AI...",
        "Launching browser...",
        "Navigating to test URL...",
        "Filling inputs...",
        "Clicking login button...",
        "Capturing screenshot...",
        "Test completed successfully!"
    ];
    const demoInterval = setInterval(() => {
        if(count < demoLogs.length) {
            appendLog(demoLogs[count]);
            count++;
        } else {
            clearInterval(demoInterval);
        }
    }, 1000);

    // Real WebSocket integration placeholder
    // const socket = new WebSocket("ws://localhost:5000/live_logs");
    // socket.onmessage = (event) => appendLog(event.data);
});
