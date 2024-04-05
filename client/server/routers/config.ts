import { Kafka, Producer } from "kafkajs";
import { ServiceModel } from "../models/service";
import { publicProcedure, router } from "../trpc";
import { z } from "zod";
import { Context } from "../_app";

const createServiceSchema = z.object({
    name: z.string(),
    description: z.string(),
    isTrain: z.boolean().default(false),
    threshold: z.number().optional(),
    coresetSize: z.number().optional(),
});

const params = z.object({
    id: z.string(),
});

const updateServiceSchema = z.object({
    params,
    body: z.object({
        name: z.string().optional(),
        description: z.string().optional(),
        isTrain: z.boolean().optional(),
        threshold: z.number().optional(),
        coresetSize: z.number().optional(),
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
    console.log(ctx);
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
        const services = await ServiceModel.find();
        return {
            status: "OK",
            data: services,
        };
    } catch (error) {
        console.error(error);
    }
};

export const configRouter = router({
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
