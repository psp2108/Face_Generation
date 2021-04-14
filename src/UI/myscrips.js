var currentTab = 0; // Current tab is set to be the first tab (0)
showTab(currentTab); // Display the current tab

function showTab(n) {
  // This function will display the specified tab of the form...
  var x = document.getElementsByClassName("tab");
  console.log(x.length);
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
  // if (n == 1 && !validateForm()) return false;
  // Hide the current tab:
  x[currentTab].style.display = "none";
  // Increase or decrease the current tab by 1:
  currentTab = currentTab + n;
  // if you have reached the end of the form...
  if (currentTab >= x.length) {
    // ... the form gets submitted:
    document.getElementById("regForm").submit();
    return false;
  }
  // Otherwise, display the correct tab:
  showTab(currentTab);
}

function validateForm() {
  // This function deals with validation of the form fields
  var x, y, i, valid = true;
  x = document.getElementsByClassName("tab");
  y = x[currentTab].getElementsByTagName("input");
  // A loop that checks every input field in the current tab:
  console.log(document.getElementsByName("5_o_clock_shadow")[0].checked);
  for (i = 0; i < y.length; i++) {
    // If a field is empty...
    if (y[i].value == "") {
      // add an "invalid" class to the field:
      y[i].className += " invalid";
      // and set the current valid status to false
      valid = false;
    }
  }
  // If the valid status is true, mark the step as finished and valid:
  if (valid) {
    document.getElementsByClassName("step")[currentTab].className += " finish";
  }
  return valid; // return the valid status

    // var arr1 = [
  //   '5_o_clock_shadow', 'bags_under_eyes', 'big_lips', 'big_nose', 'chubby', 'double_chin', 'goatee', 
  //   'heavy_makeup', 'high_cheekbones', 'male', 'mustache', 'narrow_eyes', 'no_beard', 'oval_face', 
  //   'pale_skin', 'pointy_nose', 'rosy_cheeks', 'sideburns', 'smiling', 'straight_hair', 'wavy_hair', 
  //   'young', 'hair_color', 'hair_size', 'combine_eyebrow'
  // ]

  // for(var i=0; i<arr1.length; i++){
  //   temp = document.getElementById(arr1[i]);
  //   if (temp){
  //     if (temp.checked){
  //       arr1[i] += ("=" + 1);
  //     }
  //     else{
  //       arr1[i] += ("=" + 0);
  //     }    
  //     console.log(arr1[i]);
  //   }
  //   else{
  //     console.log(arr1[i] + "element does not exist")
  //   }
  // }

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