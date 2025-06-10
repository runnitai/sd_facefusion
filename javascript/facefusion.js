// Simplified JavaScript - removed hacky status polling since we now use Gradio's built-in progress
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