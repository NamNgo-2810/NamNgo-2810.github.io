let id = location.search.slice(1);
let content = document.getElementsByClassName("content")[0];
if (id < data.length) {
    let dish = data[id-1];
    content.insertAdjacentHTML("afterbegin", 
        `<div class="dish">
        <div class="intro">
            <a>${dish.nameFood}</a>
            <img src=${dish.img} alt="">
        </div>

        <div class="recipe">${dish.content}</div>
    </div>`);
}

else {
    content.insertAdjacentHTML("afterbegin",
    `<input id="input-form" type="text">`);
}