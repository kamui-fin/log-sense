import { Service } from "../ServiceCard";
import { TextInput, Button } from "@mantine/core";
import { useState } from "react";
import z from 'zod';
import { useForm } from "@mantine/form";
import { zodResolver } from "@mantine/form";
import { trpc } from "../../../../utils/trpc";
import classes from './InputValidation.module.css';
import { Tooltip, Center, Text, rem } from "@mantine/core";
import { IconAlertTriangle} from "@tabler/icons-react";



interface RegexTab {
    service: Service;
}

type UpdateRegexInput = z.TypeOf<typeof updateRegexSchema>;

const updateRegexSchema = z.array(
    z.object({
        pattern: z.string(),
        replacement: z.string(),
    })
).default([]);

export const RegexTab = ({ service }: RegexTab) => {
    const [inputValues, setInputValues] = useState(service.regex_subs || []);
    const [patternsValid, setPatternsValid] = useState(true);
    
    const form = useForm<UpdateRegexInput>({
        initialValues: [],
        validate: zodResolver(updateRegexSchema),
    });
  
    const handleAddField = () => {
      setInputValues([...inputValues, { pattern: '', replacement: '' }]);
    };
  
    const handleSubChange = (index: number, value: string) => {
      const newInputValues = [...inputValues];
      newInputValues[index].replacement = value;
      setInputValues(newInputValues);
    };
  
    const handleRegChange = (index: number, value: string) => {
      const newInputValues = [...inputValues];
      newInputValues[index].pattern = value;
      setInputValues(newInputValues);
    };
  
    const handleDeleteField = (index: number) => {
      const newInputValues = inputValues.filter((_, i) => i !== index);
      setInputValues(newInputValues);
      updateService({
        params: {id: String(service._id)},
        body: {'regex_subs': newInputValues}
    })
    };

    const utils = trpc.useUtils();
    const { mutate: updateService } = trpc.services.updateService.useMutation({
        onSuccess() {
            utils.services.getServices.invalidate();
        },
    });
  
    const handleSubmit = () => {
        const filteredInputVals = inputValues.filter(obj => !Object.values(obj).every(value => value === ''));
        const emptyPattern = filteredInputVals.find(item => item.pattern === '');

        if(emptyPattern){
            setPatternsValid(false);

        }
        else{
            setPatternsValid(true)
            setInputValues(filteredInputVals);
            updateService({
                params: {id: String(service._id)},
                body: {'regex_subs': filteredInputVals}
            })
        }
    };

    const rightSection = (
        <Tooltip
            label={`Pattern can't be empty.`}
            position="top-end"
            withArrow
            transitionProps={{ transition: 'pop-bottom-right' }}
        >
        <Text component="div" c="dimmed" style={{ cursor: 'help' }}>
            <Center>
                <IconAlertTriangle
                    stroke={1.5}
                    style={{ width: rem(18), height: rem(18) }}
                    className={classes.icon}
                />
            </Center>
        </Text>
        </Tooltip>
    );
  
    return (
      <div className="grid grid-cols-1 gap-6">
        <div>
            {(inputValues.length == 0) ? <div style={{'margin':'10px'}}>You have no Regex.</div> :
                (inputValues.map((value, index) => (
                    <div key={index}>
                      <TextInput
                        label={`Sub ${index + 1}`}
                        value={value.replacement}
                        onChange={(event) => handleSubChange(index, event.target.value)}
                        style={{ margin: '16px' }}
                      />
                      <TextInput
                        label={`Regex ${index + 1}`}
                        description={(value.pattern.length > 0) ? '' : `ERROR: Pattern can't be empty, this SUB won't be saved.`}
                        value={value.pattern}
                        onChange={(event) => handleRegChange(index, event.target.value)}
                        classNames={{ input: ((value.pattern.length > 0) ? classes.valid : classes.invalid) }}
                        rightSection={(value.pattern.length > 0) ? '' : rightSection}
                        style={{ margin: '16px' }}
                      />
                      <Button
                        onClick={() => handleDeleteField(index)}
                        style={{ margin: '16px' }}
                      >
                        Delete -
                      </Button>
                    </div>
                  )))
            }
            <Button onClick={handleAddField} style={{ margin: '16px' }}>
                Add +
            </Button>
            <Button onClick={handleSubmit} variant="outline">
              Save
            </Button>
        </div>
      </div>
    );
  };
  