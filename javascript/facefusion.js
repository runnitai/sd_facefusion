let ffProgressTimeout = null;
let stoppedCount = 0;

function start_status() {
    stoppedCount = 0;
    ffProgressTimeout = setTimeout(recheck_status, 2000);
}

function stop_status() {
    clearTimeout(ffProgressTimeout);
}

function recheck_status() {
    let status_element = gradioApp().getElementById("statusDiv");
    if (!status_element) {
        console.log("Can't find the status element.");
        return;
    }
    let btn = gradioApp().getElementById("ff3_check_status");
    if (!btn) {
        console.log("Can't find the button.");
        return;
    }
    btn.click();
    let started = status_element.dataset.started === 'true';
    if (!started) {
        ffProgressTimeout = setTimeout(recheck_status, 2000);
    } else {
        await_progress_stop();
    }
}

function await_progress_stop() {
    clearTimeout(ffProgressTimeout);
    let status_element = gradioApp().getElementById("statusDiv");
    if (!status_element) {
        console.log("Can't find the status element.");
        return;
    }
    console.log("Data-started value: ", status_element.dataset.started);
    let started = status_element.dataset.started === 'true';
    if (started) {
        let btn = gradioApp().getElementById("ff3_check_status");
        if (!btn) {
            console.log("Can't find the button.");
            return;
        }
        btn.click();
        ffProgressTimeout = setTimeout(await_progress_stop, 1000);
    } else {
        console.log("Job stopped.");
    }
}
function get_selected_row() {
    console.log("Incoming arguments:", arguments);
    let selected = document.querySelector(".selectRow.selected");
    let res = [-1, -1];
    if (selected) {
        let rId = selected.id.replace("row", "");
        res = [rId, rId];
    }
    console.log("Selected row: ", res);
    return res;
}

document.addEventListener("DOMContentLoaded", function () {
    document.addEventListener("click", function (e) {
        // If the target is a <td> element, select the parent if it has the class "selectRow"
        // else, do nothing.
         if (e.target.tagName === "TD") {
             // Deselect all rows

            if (e.target.parentElement.classList.contains("selectRow")) {
                let rows = document.querySelectorAll(".selectRow");
                let selected = e.target.parentElement.classList.contains("selected");
                for (let i = 0; i < rows.length; i++) {
                    rows[i].classList.remove("selected");
                }
                if (!selected) {
                    e.target.parentElement.classList.add("selected");
                    let ff3_toggle_remove = gradioApp().getElementById("ff3_toggle_remove");
                    if (ff3_toggle_remove) {
                        ff3_toggle_remove.click();
                    } else {
                        console.log("Can't find toggle remove");
                    }
                }
            }
         }
    });
});