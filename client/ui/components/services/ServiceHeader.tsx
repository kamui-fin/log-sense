import { Button } from "@mantine/core";
import { IoMdArrowRoundBack } from "react-icons/io";

export const ServiceHeader = ({ name, goBack }) => {
    return (
        <div>
            <div className="flex gap-8 align-middle items-center">
                <Button onClick={() => goBack()}>
                    <IoMdArrowRoundBack />
                </Button>
                <h1>{name}</h1>
            </div>
        </div>
    );
};
