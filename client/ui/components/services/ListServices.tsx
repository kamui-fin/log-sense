import { queryOptions } from "@tanstack/react-query";
import { trpc } from "../../../utils/trpc";
import { ServiceCard } from "./ServiceCard";
import { getQueryKey } from "@trpc/react-query";

export const ListServices = () => {
    const { data: services } = trpc.config.getServices.useQuery(undefined, {
        staleTime: 5 * 1000,
        select: (data) => data?.data,
    });
    return (
        <div className="grid grid-cols-3 gap-6">
            {services?.map((service, idx) => {
                return <ServiceCard key={idx} service={service} />;
            })}
        </div>
    );
};
