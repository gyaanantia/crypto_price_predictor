const predict = ev =>{
  const textbox = document.getElementById("date");
  console.log(textbox.value);
  document.getElementById("results").innerHTML = `
    <h2>Predicted Price: </h2>
    <h3>output of model</h3>`;
  textbox.value = "";
}
