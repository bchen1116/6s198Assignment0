<!-- Contributed by David Bau, in the public domain -->

<template>
<div class="vectorlist">
  <div v-for="(vector, index) in vectors" class="vector">
    <input v-model="vector.text">
    <button @click="selectVector(index)">&rarr;</button>
    <button @click="deleteVector(index)">x</button>
  </div>
  <div class="operation">
  <button @click="saveVector()">Save current sample</button>
  </div>
  <div class="operation">
  <!-- TODO: Change this button to do something interesting -->
  <button @click="applyVectorMath()">Apply vector math</button>
  </div>
 <button @click="getKNN()">Find nearest font ID</button>
 <button @click="getAverage()">Find nearest average font</button>
</div>
</template>

<script>
import {Array1D, ENV} from 'deeplearn';
const math = ENV.math;
import * as VectorOperations from  '../utils/VectorOperations';

//This json file includes all of the Font IDs in our database and their 40-dimensional logits vector.
var json = require('../embeddings.json');

export default {
  props: {
    selectedSample: { },
    model: { },
    vectors: { type: Array, default: () => [ { text: "0" } ] }
  },
  methods: {
    saveVector() {
      this.selectedSample.data().then(x =>
         this.vectors.push({ text: Array.prototype.slice.call(x).join(',') })
      );
    },
    deleteVector(index) {
      this.vectors.splice(index, 1);
    },
    selectVector(index) {
      this.$emit("select", { selectedSample: this.model.fixdim(
           Array1D.new(this.vectors[index].text.split(',').map(parseFloat)))});
    },
    // TODO: Add useful vector space operations here -->
    applyVectorMath() {
      var c = window.prompt("Enter an array!");
      var inVec = c.split(',');
      var actual = [];
      if (inVec.length != 40) {
        window.alert("INVALID LENGTH!");
      } else {
      for (let i = 0; i < inVec.length; i++) {
        actual.push(parseFloat(inVec[i]));
      }
      this.$emit("select", { selectedSample:
           math.add(this.selectedSample, this.model.fixdim(
               Array1D.new(actual))) } )
      }
    },

    //TODO: Implement getKNN to output the font ID of the nearest neighbor
    getKNN() {
      this.selectedSample.data().then(function(x){
        var closestFontId = -1;
        var closestDistance = Number.MAX_VALUE;
        const normX = VectorOperations.norm(x);
        if (normX === 0){
          for (var i = 0; i < json.length; i++){
            const normCurrent = VectorOperations.norm(json[i])
            //console.log(normX + " compared to "+ normCurrent);
            if (normCurrent < closestDistance) {
              closestDistance = normCurrent
              closestFontId = i
            }
          }
        } else {
          for (var i = 0; i < json.length; i++){
            const currentVector = json[i]
            const dotProduct = VectorOperations.dotProduct(x, currentVector)
            const normCurrent = VectorOperations.norm(currentVector)
            const cosine = dotProduct/(normX*normCurrent)
            //console.log(normX + " compared to "+ normCurrent + ": "+cosine);

            if (cosine < closestDistance){
              closestDistance = cosine
              closestFontId = i
            }
          }
        }
        if (closestFontId !== -1) {
          window.alert("Closest Font is " + closestFontId)
        } else {
          window.alert("There is no closest font for some reason")
        }
      });
    },
    getAverage() {
      // average is determined to be the font with all of its values closest to 0.
      // We will use a squaring approach, similar to a dot product, to determine this.
      // We want to use squared numbers in order to take into account only distances
      let final = -1;
      let difference = Number.MAX_VALUE;
      for (let i = 0; i < json.length; i++) {
        let d = 0;
        for (let j = 0; j<json[i].length; j++) {
          d += (json[i][j]*json[i][j]);
        }
        if (d < difference) {
          difference = d;
          final = i;
        }
      }
      if (final !== -1) {
        window.alert("Average font is: " + final);
      } else {
        window.alert("Something went wrong!");
      }
    },
  },
  watch: {
    model: function(val) {
      for (let i = 0; i < this.vectors.length; ++i) {
        let arr = this.vectors[i].text.split(',');
        if (arr.length > this.model.dimensions) {
            arr = arr.slice(0, this.model.dimensions);
        }
        while (arr.length < this.model.dimensions) {
            arr.push('0');
        }
        this.vectors[i].text = arr.join(',');
      }
    }
  },
}
</script>

<style scoped>
.vector, .operation {
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  white-space: nowrap;
}

</style>
