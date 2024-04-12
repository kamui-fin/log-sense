import { prop, getModelForClass } from "@typegoose/typegoose";
import { RapidLog } from "./rapid_log";

class LogGptChunk {
    @prop({ required: true, type: Number })
    input_ids: number[];

    @prop({ required: true, type: Number })
    attention_mask: number[];

    @prop({ required: true, type: Number })
    labels: number[];
}

class GptLog {
    @prop({ required: true, type: Boolean })
    train_strategy: "finetune" | "ignore" | "pre-train" = "ignore";

    @prop({ required: true, type: Boolean })
    is_anomaly: boolean;

    @prop({ required: true, type: String })
    hash: string;

    @prop()
    chunks: LogGptChunk[];

    @prop()
    original_logs: RapidLog[];
}

const GptLogModel = getModelForClass(GptLog);

export { GptLog, GptLogModel };
