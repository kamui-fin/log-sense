import { getModelForClass, modelOptions, prop } from "@typegoose/typegoose";

class Service {
  @prop({ required: true, unique: true })
  name: string;

  @prop()
  description: string;

  @prop({ default: false })
  isTrain: boolean;

  @prop()
  threshold: number;

  @prop()
  coresetSize: number;
}

const ServiceModel = getModelForClass(Service);

export { Service, ServiceModel };