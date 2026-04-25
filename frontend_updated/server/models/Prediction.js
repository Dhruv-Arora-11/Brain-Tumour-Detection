const mongoose = require('mongoose')

const predictionSchema = new mongoose.Schema({
    imageName: String,
    predictedClass: String,
    confidence: Number,
    date: { type: Date, default: Date.now },
})

module.exports = mongoose.model('Prediction', predictionSchema)
