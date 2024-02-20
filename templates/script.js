// script.js

document.addEventListener("DOMContentLoaded", function () {
    const submitButton = document.querySelector("button");

    submitButton.addEventListener("mouseover", function () {
        submitButton.classList.add("animate");
    });

    submitButton.addEventListener("mouseout", function () {
        submitButton.classList.remove("animate");
    });
});
