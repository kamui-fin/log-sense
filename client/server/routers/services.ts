import { Kafka, Producer } from "kafkajs";
import { ServiceModel } from "../models/service";
import { publicProcedure, router } from "../trpc";
import { z } from "zod";
import { Context } from "../_app";

const createServiceSchema = z.object({
    name: z.string(),
    description: z.string(),
    is_train: z.boolean().default(false),
    threshold: z.number().default(-470.2),
    coreset_size: z.number().default(2),
    enable_trace: z.boolean().default(false),
    trace_regex: z.string().optional(),
    top_k: z.number().default(80),
    max_pretrain: z.number().default(10_000),
    lr_pretraining: z.number().default(1e-4),
    lr_finetuning: z.number().default(1e-6),
    train_batch_size: z.number().default(16),
    num_episodes: z.number().default(10),
    num_epochs: z.number().default(10),
    vocab_size: z.number().default(500),
    regex_subs: z
        .array(
            z.object({
                pattern: z.string(),
                replacement: z.string(),
            })
        )
        .default([]),
});

const params = z.object({
    id: z.string(),
});

const updateServiceSchema = z.object({
    params,
    body: z.object({
        name: z.string().optional(),
        description: z.string().optional(),
        is_train: z.boolean().optional(),
        threshold: z.number().optional(),
        coreset_size: z.number().optional(),
        enable_trace: z.boolean().optional(),
        trace_regex: z.string().optional(),
        top_k: z.number().optional(),
        max_pretrain: z.number().optional(),
        lr_pretraining: z.number().optional(),
        lr_finetuning: z.number().optional(),
        train_batch_size: z.number().optional(),
        num_episodes: z.number().optional(),
        num_epochs: z.number().optional(),
        vocab_size: z.number().optional(),
        regex_subs: z
            .array(
                z.object({
                    pattern: z.string(),
                    replacement: z.string(),
                })
            )
            .optional(),
    }),
});

type ParamsInput = z.TypeOf<typeof params>;
type CreateServiceInput = z.TypeOf<typeof createServiceSchema>;
type UpdateServiceInput = z.TypeOf<typeof updateServiceSchema>;

const updateKafka = (producer: Producer) => {
    const kafkaMessage = {
        value: JSON.stringify({ type: "config-change" }),
    };
    producer.send({
        topic: "config-change",
        messages: [kafkaMessage],
    });
};

const createServiceController = async ({
    input,
    ctx,
}: {
    input: CreateServiceInput;
    ctx: Context;
}) => {
    try {
        const createdService = new ServiceModel(input).save();
        updateKafka(ctx.kafkaProducer);
        return {
            status: "OK",
            data: createdService,
        };
    } catch (error) {
        console.error(error);
    }
};

const updateServiceController = async ({
    paramsInput,
    input,
    ctx,
}: {
    paramsInput: ParamsInput;
    input: UpdateServiceInput["body"];
    ctx: Context;
}) => {
    try {
        const updatedService = await ServiceModel.updateOne(
            { _id: paramsInput.id },
            input
        );
        updateKafka(ctx.kafkaProducer);
        return {
            status: "OK",
            data: updatedService,
        };
    } catch (error) {
        console.error(error);
    }
};

const deleteServiceController = async ({
    paramsInput,
    ctx,
}: {
    paramsInput: ParamsInput;
    ctx: Context;
}) => {
    try {
        const deletedService = await ServiceModel.deleteOne({
            _id: paramsInput.id,
        });
        updateKafka(ctx.kafkaProducer);
        return {
            status: "OK",
            data: deletedService,
        };
    } catch (error) {
        console.error(error);
    }
};

const getServicesController = async () => {
    try {
        const services = (await ServiceModel.find()).map((service) => ({
            ...service.toObject(),
            _id: service._id.toString(),
        }));
        return {
            status: "OK",
            data: services,
        };
    } catch (error) {
        console.error(error);
    }
};

export const serviceRouter = router({
    createService: publicProcedure
        .input(createServiceSchema)
        .mutation(({ input, ctx }) => createServiceController({ input, ctx })),
    updateService: publicProcedure
        .input(updateServiceSchema)
        .mutation(({ input, ctx }) =>
            updateServiceController({
                paramsInput: input.params,
                input: input.body,
                ctx,
            })
        ),
    deleteService: publicProcedure
        .input(params)
        .mutation(({ input, ctx }) =>
            deleteServiceController({ paramsInput: input, ctx })
        ),
    getServices: publicProcedure.query(() => getServicesController()),
});
