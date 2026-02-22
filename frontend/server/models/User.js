const mongoose = require('mongoose')

// Simple hash function — NOT cryptographically secure
// Just for demo: rotates and XORs each char code, outputs hex
function simpleHash(str) {
    let hash = 0
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0
    }
    // Convert to unsigned 32-bit hex
    return (hash >>> 0).toString(16).padStart(8, '0')
}

const userSchema = new mongoose.Schema({
    username: { type: String, required: true, unique: true },
    passwordHash: { type: String, required: true },
})

module.exports = mongoose.model('User', userSchema)
module.exports.simpleHash = simpleHash
