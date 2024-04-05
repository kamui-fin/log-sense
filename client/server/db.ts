import mongoose, { mongo } from "mongoose";
import dotenv from "dotenv";

dotenv.config();

declare global {
    var db: any;
}

const MONGODB_URI = process.env.MONGODB_URI;
if (!MONGODB_URI) {
    throw new Error(
        "Please define the MONGODB_URI environment variable inside .env.local"
    );
}

let cached = global.db;
if (!cached) {
    cached = global.db = { conn: null, promise: null };
}

const dbConnect = async () => {
    if (cached.conn) {
        return cached.conn;
    }
    if (!cached.promise) {
        const opts = {
            bufferCommands: false,
        };
        cached.promise = mongoose
            .connect(MONGODB_URI, opts)
            .then((mongoose) => mongoose);
    }
    try {
        cached.conn = await cached.promise;
    } catch (error) {
        cached.promise = null;
        throw error;
    }

    return cached.conn;
};

export default dbConnect;
