function dotProduct(vector1, vector2){
  if (vector1.length !== vector2.length){
    throw new Error("Wrong dimensions!")
  }
  var totalSum = 0.0;
  for (var i = 0; i < vector1.length; i++){
    totalSum += (new Number(vector1[i]))*(new Number(vector2[i]))
  }
  return totalSum
}

function norm(vector){
  const square = dotProduct(vector, vector)
  return Math.sqrt(square)
}

export {dotProduct, norm}
