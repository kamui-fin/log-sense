import { prop, getModelForClass, mongoose } from "@typegoose/typegoose";

class RapidLog {
  _id: mongoose.Types.ObjectId;

  @prop({ required: true, type: String })
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

  // By default, we assume all logs sitting in the database are anomalies
  @prop({ required: true, type: Boolean })
  is_anomaly: boolean = true;

  @prop({ required: false, type: Number })
  score: number;

  @prop({ required: true, type: Boolean })
  train_strategy: "finetune" | "ignore" | "pre-train" = "ignore";

  @prop({ required: false, type: Boolean, default: true })
  prompt_user: boolean;
}

const RapidLogModel = getModelForClass(RapidLog);

export { RapidLog, RapidLogModel };
