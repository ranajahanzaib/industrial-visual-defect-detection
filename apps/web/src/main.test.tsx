// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main";

function mount() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );
}
function response(body: unknown, status = 200) {
  return { ok: status < 400, status, json: async () => body } as Response;
}

describe("Industrial Visual Defect Detection workbench navigation", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/runs")) return response([]);
        if (url.endsWith("/api/datasets"))
          return response({ available: false, categories: [] });
        return response({}, 404);
      }),
    );
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows an honest empty state when no trained run exists", async () => {
    mount();
    expect(await screen.findByText("0 AVAILABLE")).toBeTruthy();
    expect(screen.getByText("Awaiting image input")).toBeTruthy();
    expect(screen.getByText("Select a run")).toBeTruthy();
  });

  it("opens evaluation and dataset empty states from the navigation", async () => {
    mount();
    fireEvent.click(await screen.findByRole("button", { name: "Evaluation" }));
    expect(
      screen.getByText(
        "Choose a run with a completed held-out evaluation report.",
      ),
    ).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Dataset explorer" }));
    expect(
      await screen.findByText(/Opening this explorer does not download data/),
    ).toBeTruthy();
    expect(screen.getByText("DATA NOT INSTALLED")).toBeTruthy();
  });
});
