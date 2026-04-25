const express = require('express')
const multer = require('multer')
const cors = require('cors')
const path = require('path')
const { execSync } = require('child_process')

const app = express()
const PORT = 5000

app.use(cors())
app.use(express.json())

let Prediction = null
let User = null
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/neuroscan'

const mongoose = require('mongoose')
mongoose.set('strictQuery', false)
mongoose
    .connect(MONGO_URI, { serverSelectionTimeoutMS: 5000 })
    .then(() => {
        console.log('MongoDB connected')
        Prediction = require('./models/Prediction')
        User = require('./models/User')
    })
    .catch((err) => {
        console.log('MongoDB not available — running without database')
    })

const upload = multer({ dest: 'uploads/' })

// ── Auth routes ──────────────────────────────────────────

app.post('/api/register', async (req, res) => {
    const { username, password } = req.body
    if (!username || !password) return res.status(400).json({ error: 'Username and password required' })
    if (!User) return res.status(503).json({ error: 'Database not available' })

    const { simpleHash } = require('./models/User')
    const existing = await User.findOne({ username })
    if (existing) return res.status(409).json({ error: 'Username already taken' })

    const user = await User.create({ username, passwordHash: simpleHash(password) })
    res.json({ success: true, user: { _id: user._id, username: user.username } })
})

app.post('/api/login', async (req, res) => {
    const { username, password } = req.body
    if (!username || !password) return res.status(400).json({ error: 'Username and password required' })
    if (!User) return res.status(503).json({ error: 'Database not available' })

    const { simpleHash } = require('./models/User')
    const user = await User.findOne({ username })
    if (!user || user.passwordHash !== simpleHash(password)) {
        return res.status(401).json({ error: 'Invalid username or password' })
    }
    res.json({ success: true, user: { _id: user._id, username: user.username } })
})

// ── Predict routes ───────────────────────────────────────

app.post('/api/predict', upload.single('image'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'No image uploaded' })

    const imagePath = path.resolve(req.file.path)
    const scriptPath = path.resolve(__dirname, 'predict.py')

    let output
    try {
        output = execSync(`python3 "${scriptPath}" "${imagePath}"`, {
            encoding: 'utf-8',
            timeout: 60000,
        })
    } catch {
        output = execSync(`python "${scriptPath}" "${imagePath}"`, {
            encoding: 'utf-8',
            timeout: 60000,
        })
    }

    try {
        const result = JSON.parse(output.trim())

        if (Prediction) {
            await Prediction.create({
                imageName: req.file.originalname,
                predictedClass: result.className,
                confidence: result.confidence,
            })
        }

        res.json(result)
    } catch (err) {
        res.status(500).json({ error: 'Prediction failed: ' + err.message })
    }
})

app.get('/api/history', async (req, res) => {
    if (!Prediction) return res.json([])
    const history = await Prediction.find().sort({ date: -1 }).limit(10)
    res.json(history)
})

app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`))
