let names_for_search = [];
function show(dishes) {
    remove();
    let content = document.getElementsByClassName("content")[0];
    for (let dish of dishes) {
        let name = dish.nameFood;
        let image = dish.img;
        let intro = dish.intro;
        content.insertAdjacentHTML("beforeend", `
        <div class="dish">
            <a class="food-name" href="specific_dish.html?${dish.id}" rel="bookmark">${name}</a>
            <input class="chosen" type="checkbox" style="visibility: hidden">
            <input class="rename" type="text" style="visibility: hidden">
            <button class="update" style="visibility: hidden">Rename</button> 
            <div class="food-img">
                <img src=${image} alt="">
            </div>
            <div class="dish-content">${intro}</div>
        </div>
        `);
        names_for_search.push(name);
    }
}
function remove() {
    let content = document.getElementsByClassName("content")[0];
    content.innerHTML = '';
}
function findType(type) {
    let dishes = []
    for (let dish of data) {
        for (let typeOfDish of dish.type) {
            if (typeOfDish == type) {
                dishes.push(dish);
            }
        }
    }
    return dishes;
}

let adminMode = false;
let loged_in = document.getElementById("enter");
loged_in.addEventListener("click", function() {
    adminMode = true;
    showAdminMode();
    nameUpdate();
});
function showAdminMode() {
    let chosen = document.getElementsByClassName("chosen");
    let rename = document.getElementsByClassName("rename");
    let rename_btn = document.getElementsByClassName("update");
    let length = rename.length;
    for (let i = 0; i < length; i++) {
        chosen[i].style.visibility = "visible";
        rename[i].style.visibility = "visible";
        rename_btn[i].style.visibility = "visible";
        rename_btn[i].style.borderRadius = "5px";
    }
    remove_btn.style.display = "block";
    remove_btn.style.zIndex = "1";
    add_btn.style.display = "block";
}

function remove_dishes() {
    let choosed = {};
    let chosen = document.getElementsByClassName("chosen");
    let length = chosen.length;
    for (let i = 0; i < length; i++) {
        if (chosen[i].checked == true) {
            choosed[i] = true;
        }
    } 
    remove();   
    let newData = [];
    for (let i = 0; i < length; i++) {
        if (choosed[i] == null) {
            newData.push(data[i]);
        }
    }
    data = newData;
    show(location.search.slice(1) > 0 ? findType(location.search.slice(1)) : data);
}
function nameUpdate() {
    let foodName = document.getElementsByClassName("food-name");
    let rename_btn = document.getElementsByClassName("update");
    let newName = document.getElementsByClassName("rename");
    for (let i = 0; i < foodName.length; i++) {
        rename_btn[i].addEventListener("click", function() {
            console.log("Pressed");
            if (newName[i].value.length > 0) {
                foodName[i].innerHTML = newName[i].value;
            }
        });
        
    }
}
let kindOfDish = location.search.slice(1);
if (adminMode) {
    showAdminMode();
    
}
else {
    if (kindOfDish.length > 0) show(findType(kindOfDish));
    else show(data);
}
let remove_btn = document.getElementById("remove-dishes");
remove_btn.addEventListener("click", function() {
    remove_dishes();
});
let start_search = false;
function autocomplete(inp, arr) {
    var currentFocus;
    inp.addEventListener("input", function() {
        var a, b, i, val = this.value;
        closeAllLists();
        if (!val) { return false;}
        currentFocus = -1;
        a = document.createElement("DIV");
        a.setAttribute("id", this.id + "autocomplete-list");
        a.setAttribute("class", "autocomplete-items");
        
        this.parentNode.appendChild(a);

        for (i = 0; i < arr.length; i++) {
            
            if (arr[i].substr(0, val.length).toUpperCase() == val.toUpperCase()) {
                b = document.createElement("DIV");
                b.innerHTML = "<strong>" + arr[i].substr(0, val.length) + "</strong>";
                b.innerHTML += arr[i].substr(val.length);
                b.innerHTML += "<input type='hidden' value='" + arr[i] + "'>";
                b.addEventListener("click", function() {
                    inp.value = this.getElementsByTagName("input")[0].value;
                    closeAllLists();
                });
                a.appendChild(b);
            }
        }
    });
    inp.addEventListener("keydown", function(e) {
        var x = document.getElementById(this.id + "autocomplete-list");
        if (x) x = x.getElementsByTagName("div");
        if (e.keyCode == 40) {
            currentFocus++;
            addActive(x);
        } 
        else if (e.keyCode == 38) { 
            currentFocus--;
            addActive(x);
        } 
        else if (e.keyCode == 13) {
            e.preventDefault();
            if (currentFocus > -1) {
                if (x) x[currentFocus].click();
            }
        }
    });
    function addActive(x) {
        if (!x) return false;
        removeActive(x);
        if (currentFocus >= x.length) currentFocus = 0;
        if (currentFocus < 0) currentFocus = (x.length - 1);
        x[currentFocus].classList.add("autocomplete-active");
    }
    function removeActive(x) {
        for (var i = 0; i < x.length; i++) {
            x[i].classList.remove("autocomplete-active");
        }
    }
    function closeAllLists(elmnt) {
        var x = document.getElementsByClassName("autocomplete-items");
        for (var i = 0; i < x.length; i++) {
            if (elmnt != x[i] && elmnt != inp) {
                x[i].parentNode.removeChild(x[i]);
            }
        }
    }
    document.addEventListener("click", function (e) {
        closeAllLists(e.target);
        start_search = true;
    });
}
let search_mode = false;
let search_btn = document.getElementById("searching");
function search() {
    let input = document.getElementById("hiding-search").value;
    let id;
    for (let i = 0; i < data.length; i++) {
        
        if (data[i].nameFood == input) {
            id = i+1;
            break;
        }
    }
    location = `http://127.0.0.1:5500/Kitchies/specific_dish.html?${id}`;
}

search_btn.addEventListener("click", () => {
    autocomplete(document.getElementById("hiding-search"), names_for_search);
    if (start_search) {
        search_btn.addEventListener("click", search());
    }
});

function add() {
    location = `http://127.0.0.1:5500/Kitchies/specific_dish.html?${data.length+1}`;
}
let add_btn = document.getElementById("add-dish");
add_btn.addEventListener("click", function() {add()});