import {
  getModelForClass,
  modelOptions,
  mongoose,
  prop,
} from "@typegoose/typegoose";

class Service {
  _id!: mongoose.Types.ObjectId;

  @prop({ required: true, unique: true, type: String })
  name!: string;

  @prop({ type: String })
  description!: string;

  @prop({ default: false, type: Boolean })
  is_train!: boolean;

  @prop({ type: Number })
  threshold!: number;

  @prop({ type: Number })
  coreset_size!: number;

  @prop({ type: Boolean, default: false })
  enable_trace!: boolean;

  @prop({ type: String })
  trace_regex?: string;

  @prop({ type: Number, default: 80 })
  top_k!: number;

  @prop({ type: Number, default: 10_000 })
  max_pretrain!: number;

  @prop({ type: Number, default: 512 })
  context_size!: number;

  @prop({ type: Number, default: 1e-4 })
  lr_pretraining!: number;

  @prop({ type: Number, default: 1e-6 })
  lr_finetuning!: number;

  @prop({ type: Number, default: 16 })
  train_batch_size!: number;

  @prop({ type: Number, default: 10 })
  num_episodes!: number;

  @prop({ type: Number, default: 10 })
  num_epochs!: number;

  @prop({ type: Number, default: 500 })
  vocab_size!: number;
}

const ServiceModel = getModelForClass(Service);

export { Service, ServiceModel };
