import React from "react";
import { QueryClient, QueryClientProvider } from "react-query";

const client = new QueryClient();
interface Props {
  children: React.ReactNode;
}
const ReactQueryProvider: React.FunctionComponent<Props> = ({ children }) => {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
};

export default ReactQueryProvider;
