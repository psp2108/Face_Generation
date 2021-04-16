var currentTab = 0; // Current tab is set to be the first tab (0)
showTab(currentTab); // Display the current tab
var api = "http://127.0.0.1/";

function showTab(n) {
  // This function will display the specified tab of the form...
  var x = document.getElementsByClassName("tab");
  if(x.length > 0)
    x[n].style.display = "block";
  //... and fix the Previous/Next buttons:
  if (n == 0) {
    document.getElementById("prevBtn").style.display = "none";
  } else {
    document.getElementById("prevBtn").style.display = "inline";
  }
  if (n == (x.length - 1)) {
    document.getElementById("nextBtn").innerHTML = "Submit";
  } else {
    document.getElementById("nextBtn").innerHTML = "Next";
  }
  //... and run a function that will display the correct step indicator:
  fixStepIndicator(n)
}

function nextPrev(n) {
  // This function will figure out which tab to display
  var x = document.getElementsByClassName("tab");
  // Exit the function if any field in the current tab is invalid:

  // Hide the current tab:
  if (document.getElementById("nextBtn").innerHTML == "Submit" && n==1){
    validateForm();
    
  }
  else{
    x[currentTab].style.display = "none";
    // Increase or decrease the current tab by 1:
    currentTab = currentTab + n;
  }
  showTab(currentTab);
}



function validateForm() {
    var arr1 = [
    '5_o_clock_shadow', 'bags_under_eyes', 'big_lips', 'big_nose', 'chubby', 'double_chin', 'goatee', 
    'heavy_makeup', 'high_cheekbones', 'male', 'mustache', 'narrow_eyes', 'no_beard', 'oval_face', 
    'pale_skin', 'pointy_nose', 'rosy_cheeks', 'sideburns', 'smiling', 'straight_hair', 'wavy_hair', 
    'young', 'hair_color', 'hair_size', 'combine_eyebrow'
  ]

  var arr2 = [
    'rv0=0.643303300', 'rv1=0.956696700', 'rv2=0.643303300', 'rv3=0.643303300', 'rv4=0.643303300', 'rv5=0.643303300', 'rv6=0.1356696700'
]

  var exception_array = [
    "hair_size","hair_color","combine_eyebrow","male"
  ]

  for(var i=0; i<arr1.length; i++){
    temp = document.getElementById(arr1[i]);
    if (temp){
      if (exception_array.includes(arr1[i])){
        arr1[i]+= ("="+handle_exception(arr1[i]));
      }
      else{
        if (temp.checked){
          arr1[i] += ("=" + 1);
        }
        else{
          arr1[i] += ("=" + 0);
        }    
        //console.log(arr1[i]);
      }
      
    }
    else{
      arr1[i] += ("=" + 0);
      console.log(arr1[i] + "element does not exist")
    }
  }

  var all_attribs = arr1.join('&')

  console.log(all_attribs);
  document.getElementById('preview_img').src= api + "get_image?" + all_attribs;
  document.getElementById('prv_img_get_id').style.visibility="visible";
}

function fixStepIndicator(n) {
  // This function removes the "active" class of all steps...
  var i, x = document.getElementsByClassName("step");
  for (i = 0; i < x.length; i++) {
    x[i].className = x[i].className.replace(" active", "");
  }
  //... and adds the "active" class on the current step:
  x[n].className += " active";
}


function handle_exception(excep) {
  console.log("FN CALLED");
  selection = document.getElementsByName(excep);
  console.log(selection.length);
  switch (excep) {
    case "hair_size":
      if(document.getElementById("hair_size").checked)
      return 1;
      else if(document.getElementById("hair_size_medium").checked)
      return 0.5;
      else
      return 0;
      break;
    
    case "hair_color":
      if(document.getElementById("hair_color").checked)
      return 1;
      else if(document.getElementById("hair_color_gray").checked)
      return 0.25;
      else if(document.getElementById("hair_color_brown").checked)
      return 0.75;
      else
      return 0.5;
      break;

    case "combine_eyebrow":
      if(document.getElementById("combine_eyebrow").checked && document.getElementById("bushy_eyebrow").checked)
        return 0.5;
      else if(document.getElementById("combine_eyebrow").checked && document.getElementById("bushy_eyebrow").checked===false)
        return 0;
      else
        return 1;
      break;
    default:
      return 0;

    case "male":
      return document.getElementById("male").value;
  }
}


function chng_gender(val){
  document.getElementById("male").value = val;
  if(val){
    
    document.getElementById("female").style.backgroundImage = "url('female.png')";
    document.getElementById("male").style.backgroundImage = "url('bluemale.png')";
  }
  else{
    document.getElementById("female").style.backgroundImage = "url('bluefemale.png')";
    document.getElementById("male").style.backgroundImage = "url('male.png')";
  }
  
  
}

function getapi(){
  document.getElementById("id_href").href = api + "get_latest_details";
}