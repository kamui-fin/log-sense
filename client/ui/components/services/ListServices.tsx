import { trpc } from "../../../utils/trpc";
import { ServiceCard } from "./ServiceCard";

export const ListServices = () => {
    const { data: services } = trpc.config.getServices.useQuery("getServices", {
        staleTime: 5 * 1000,
        select: (data) => data?.data,
    });
    return (
        <div className="grid grid-cols-3 gap-6">
            {services?.map((service) => {
                return <ServiceCard service={service} />;
            })}
        </div>
    );
};
