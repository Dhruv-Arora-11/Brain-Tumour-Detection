const express = require('express')
const mongoose = require('mongoose')
const multer = require('multer')
const cors = require('cors')
const path = require('path')
const { execSync } = require('child_process')
const Prediction = require('./models/Prediction')

const app = express()
const PORT = 5000

app.use(cors())
app.use(express.json())

mongoose.connect('mongodb://localhost:27017/neuroscan')

const upload = multer({ dest: 'uploads/' })

app.post('/api/predict', upload.single('image'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'No image uploaded' })

    const imagePath = path.resolve(req.file.path)
    const scriptPath = path.resolve(__dirname, 'predict.py')

    try {
        const output = execSync(`python "${scriptPath}" "${imagePath}"`, {
            encoding: 'utf-8',
            timeout: 30000,
        })

        const result = JSON.parse(output.trim())

        await Prediction.create({
            imageName: req.file.originalname,
            predictedClass: result.className,
            confidence: result.confidence,
        })

        res.json(result)
    } catch (err) {
        res.status(500).json({ error: 'Prediction failed: ' + err.message })
    }
})

app.get('/api/history', async (req, res) => {
    const history = await Prediction.find().sort({ date: -1 }).limit(10)
    res.json(history)
})

app.listen(PORT, () => console.log(`Server running on port ${PORT}`))
