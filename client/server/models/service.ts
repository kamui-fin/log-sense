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
    isTrain!: boolean;

    @prop({ type: Number })
    threshold!: number;

    @prop({ type: Number })
    coresetSize!: number;

    @prop({ type: Boolean, default: false })
    enableTrace!: boolean;

    @prop({ type: String })
    traceRegex?: string;
}

const ServiceModel = getModelForClass(Service);

export { Service, ServiceModel };
