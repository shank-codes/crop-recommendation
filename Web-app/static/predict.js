
function predict() {
    let form = document.forms['form']
    let input;
    for(let i = 0; i < form.length; i++) {
        input[form.elements[i].name] = form.elements[i].value;
    }
    console.log(input);
    fetch("http://127.0.0.1:8001/predict", {
        method: "POST",
        body: JSON.stringify(input),
        headers: {
            "Content-type": "application/json"
        }
    })
    .then(response => response.json) 
    .then(json => {
        console.log(json)
        if(json["status"] === 200) {
            let head = document.getElementById("head");
            head.innerHTML = "Grow " + json["result"];
        }
    });
}