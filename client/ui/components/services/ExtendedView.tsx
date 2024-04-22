import { ServiceCardProps } from "./ServiceCard";
import { ServiceHeader } from "./ServiceHeader";
import { ListServices } from "./ListServices";
import { useState } from 'react';
import { PasswordInput, Tooltip, Center, Text, rem } from '@mantine/core';
import { IconInfoCircle } from '@tabler/icons-react';
import { useForm, zodResolver } from "@mantine/form";
import { useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { trpc } from "../../../utils/trpc";
import { Button, Modal, NumberInput, Switch, TextInput } from "@mantine/core";
import { Tabs} from '@mantine/core';
import { IconPhoto, IconMessageCircle, IconSettings } from '@tabler/icons-react';
import { GeneralTab } from "./tabs/ServiceGeneral";
import { RAPIDTab } from "./tabs/ServiceRAPID";
import { GPTTab } from "./tabs/ServiceGPT";
import { useEffect, useRef } from "react";

interface General {
    name: string;
    description: string;
    isTrain: boolean;
}

interface RAPID {
    coresetSize: number;
    threshold: number;
}

interface GPT {
    top_k: number;
    max_pretrain: number;
    vocab_size: number;
    lr_pretraining: number;
    lr_finetuning: number;
    train_batch_size: number;
    num_episodes: number;
    num_epochs: number;
    enable_trace: boolean;
}


export const ExtendedView = ( {service, onGoBack}: ServiceCardProps) => {
    let generalValues: General = {name:service.name, description: service.description, isTrain:service.isTrain};
    let rapidValues: RAPID = {coresetSize: service.coresetSize, threshold: service.threshold};
    let gptValues: GPT = {
        top_k: service.top_k, 
        max_pretrain: service.max_pretrain, 
        vocab_size: service.vocab_size, 
        lr_pretraining: service.lr_pretraining, 
        lr_finetuning: service.lr_finetuning, 
        train_batch_size: service.train_batch_size,
        num_episodes: service.num_episodes,
        num_epochs: service.num_epochs,
        enable_trace: service.enable_trace
    }

    const queryClient = useQueryClient();

    const { mutate: updateService } = trpc.config.updateService.useMutation({
        onSuccess() {
            queryClient.invalidateQueries(["getNotes"]);
        },
    });

    const updateAllValues = () => {
        const allValues = {
            ...generalValues,
            ...rapidValues,
            ...gptValues,
        };
        console.log(allValues)
        updateService({ params: { id: service._id , type: 'SERVICE'}, body: allValues });
        onGoBack();
    }

    const submitGeneral = (newGeneralValues) => {
        generalValues = newGeneralValues;
        updateAllValues()
    }

    const submitRAPID = (newRapidValues) => {
        rapidValues = newRapidValues;
        updateAllValues()
    }

    const submitGPT = (newGptValues) =>{
        gptValues = newGptValues;
        updateAllValues()
    }
    
    const iconStyle = { width: rem(12), height: rem(12) };
    return(
        <div>
            <ServiceHeader name={service.name} goBack={onGoBack}/>
            <Tabs>
                <Tabs.List>
                    <Tabs.Tab value="general">
                        General
                    </Tabs.Tab>
                    <Tabs.Tab value="rapid">
                        RAPID
                    </Tabs.Tab>
                    <Tabs.Tab value="gpt">
                        LogGPT
                    </Tabs.Tab>
                </Tabs.List>
                <Tabs.Panel value="general">
                    <GeneralTab service = {service} onGoBack = {onGoBack} onSubmitTab={submitGeneral}/>
                </Tabs.Panel>
                <Tabs.Panel value="rapid">
                    <RAPIDTab service = {service} onGoBack={onGoBack} onSubmitTab={submitRAPID}/>
                </Tabs.Panel>
                <Tabs.Panel value="gpt">
                    <GPTTab service ={service} onGoBack={onGoBack} onSubmitTab={submitGPT}/>
                </Tabs.Panel>
        </Tabs>
        </div>
    );
}