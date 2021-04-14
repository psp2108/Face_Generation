var currentTab = 0; // Current tab is set to be the first tab (0)
showTab(currentTab); // Display the current tab

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

  for(var i=0; i<arr1.length; i++){
    temp = document.getElementById(arr1[i]);
    if (temp){
      if (temp.checked){
        arr1[i] += ("=" + 1);
      }
      else{
        arr1[i] += ("=" + 0);
      }    
      console.log(arr1[i]);
    }
    else{
      arr1[i] += ("=" + 0);
      console.log(arr1[i] + "element does not exist")
    }
  }

  var all_attribs = arr1.join('&') + "&" + arr2.join('&');

  console.log(all_attribs);
  document.getElementById('preview_img').src="http://127.0.0.1/get_image?" + all_attribs;

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