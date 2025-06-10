// Minimal status checking for preview image updates
let statusCheckInterval = null;

function start_status_check() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    statusCheckInterval = setInterval(check_status, 1000);
}

// Alias for compatibility
function start_status() {
    start_status_check();
}

function stop_status_check() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

// Alias for compatibility
function stop_status() {
    stop_status_check();
}

function check_status() {
    let status_element = gradioApp().getElementById("statusDiv");
    if (!status_element) {
        return;
    }
    
    let started = status_element.dataset.started === 'true';
    if (started) {
        let btn = gradioApp().getElementById("ff3_check_status");
        if (btn) {
            btn.click();
        }
    } else {
        stop_status_check();
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