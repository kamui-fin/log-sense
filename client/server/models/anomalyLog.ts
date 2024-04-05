import { prop, getModelForClass } from "@typegoose/typegoose";

class Log {
    @prop({ required: true, unique: true, type: String })
    hash: string;

    @prop({ required: true, type: String })
    service: string;

    @prop({ required: true, type: String })
    node: string;

    @prop({ required: true, type: String })
    filename: string;

    @prop({ required: true, type: String })
    original_text: string;

    @prop({ required: true, type: String })
    cleaned_text: string;

    @prop({ required: true, type: Number })
    timestamp: number;

    @prop({ required: true, type: Number })
    score: number;
}

const LogModel = getModelForClass(Log);

export { Log, LogModel };
