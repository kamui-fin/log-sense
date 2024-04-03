import { prop, getModelForClass } from '@typegoose/typegoose';

class Log {
  @prop({ required: true, unique: true })
  hash_id: number;

  @prop({ required: true })
  service: string;

  @prop({ required: true })
  node: string;

  @prop({ required: true })
  filename: string;

  @prop({ required: true })
  originalText: string;

  @prop({ required: true })
  cleanedText: string;

  @prop({ required: true })
  timestamp: number;

  @prop({ required: true })
  score: number
}

const LogModel = getModelForClass(Log);

export { Log, LogModel };