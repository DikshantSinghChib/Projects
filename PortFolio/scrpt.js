document.addEventListener("DOMContentLoaded", function () {
    let button1 = document.querySelector(".button1");
    let button2 = document.querySelector(".button2");
    let hiddenCards = document.querySelectorAll(".p_card.hide");

    function activate1() {
        hiddenCards.forEach(card => card.style.display = "flex");
        button1.style.display = "none";
        button2.style.display = "block";
    }

    function activate2() {
        hiddenCards.forEach(card => card.style.display = "none");
        button1.style.display = "block";
        button2.style.display = "none";
    }

    button1.addEventListener("click", activate1);
    button2.addEventListener("click", activate2);
});
