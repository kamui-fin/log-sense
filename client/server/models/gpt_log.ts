import { prop, getModelForClass, modelOptions } from "@typegoose/typegoose";
import { RapidLog } from "./rapid_log";

@modelOptions({ schemaOptions: { _id: false } })
class LogGptChunk {
    @prop({ required: true, type: Array<Number> })
    input_ids!: number[];
    @prop({ required: true, type: Array<Number> })
    attention_mask!: number[];
    @prop({ required: true, type: Array<Number> })
    labels!: number[];
}

class GptLog {
    @prop({ required: true, type: String, default: "ignore" })
    train_strategy!: "finetune" | "ignore" | "pre-train";

    @prop({ required: true, type: Boolean })
    is_anomaly!: boolean;

    @prop({ required: true, type: String })
    hash!: string;

    @prop({ required: true, type: LogGptChunk })
    chunk!: LogGptChunk;

    @prop({ required: true, type: Array<RapidLog> })
    original_logs!: RapidLog[];

    @prop({ required: false, type: Boolean, default: true })
    prompt_user!: boolean;
}

const GptLogModel = getModelForClass(GptLog);

export { LogGptChunk, GptLog, GptLogModel };
