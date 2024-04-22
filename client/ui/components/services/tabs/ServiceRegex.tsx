import { Service } from "@/components/server/models/service";
import { TextInput, Button } from "@mantine/core";
import { useState } from "react";
import z from 'zod';
import { useForm } from "@mantine/form";
import { serviceRouter } from "@/components/server/routers/services";
import { zodResolver } from "@mantine/form";
import { trpc } from "../../../../utils/trpc";

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
    
    const form = useForm<UpdateRegexInput>({
        initialValues: [],
        validate: zodResolver(updateRegexSchema),
    });
  
    const handleAddField = () => {
      setInputValues([...inputValues, { pattern: '', replacement: '' }]);
    };
  
    const handleSubChange = (index, value) => {
      const newInputValues = [...inputValues];
      newInputValues[index].replacement = value;
      setInputValues(newInputValues);
    };
  
    const handleRegChange = (index, value) => {
      const newInputValues = [...inputValues];
      newInputValues[index].pattern = value;
      setInputValues(newInputValues);
    };
  
    const handleDeleteField = (index) => {
      const newInputValues = inputValues.filter((_, i) => i !== index);
      setInputValues(newInputValues);
      updateService({
        params: {id: service._id},
        body: {'regex_subs': newInputValues}
    })
    };

    const utils = trpc.useUtils();
    const { mutate: updateService } = trpc.services.updateService.useMutation({
        onSuccess() {
            utils.services.getServices.invalidate();
        },
    });

    const updateVals = () => {
        updateService({
            params: {id: service._id},
            body: {'regex_subs': inputValues}
        })
    };
  
    const handleSubmit = () => {
        const filteredInputVals = inputValues.filter(obj => !Object.values(obj).every(value => value === ''))
        setInputValues(filteredInputVals);
        updateService({
            params: {id: service._id},
            body: {'regex_subs': filteredInputVals}
        })
    };
  
    return (
      <div className="grid grid-cols-1 gap-6">
        <div>
          {/* <form onSubmit={form.onSubmit((values) => console.log(values))}> */}
            {inputValues.map((value, index) => (
              <div key={index}>
                <TextInput
                  label={`Sub ${index + 1}`}
                  value={value.replacement}
                  onChange={(event) => handleSubChange(index, event.target.value)}
                  style={{ margin: '16px' }}
                />
                <TextInput
                  label={`Regex ${index + 1}`}
                  value={value.pattern}
                  onChange={(event) => handleRegChange(index, event.target.value)}
                  style={{ margin: '16px' }}
                />
                <Button
                  onClick={() => handleDeleteField(index)}
                  style={{ margin: '16px' }}
                >
                  Delete Field
                </Button>
              </div>
            ))}
            <Button onClick={handleAddField} style={{ margin: '16px' }}>
                Add Field
            </Button>
            <Button onClick={handleSubmit} variant="outline">
              Submit
            </Button>
          {/* </form> */}
        </div>
      </div>
    );
  };
  