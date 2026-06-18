import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppLayout } from "@/components/app-layout";
import FontPreviewerPage from "@/pages/font-previewer";
import EvaluationDashboard from "@/pages/evaluation-dashboard";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<FontPreviewerPage />} />
          <Route path="evaluation" element={<EvaluationDashboard />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;