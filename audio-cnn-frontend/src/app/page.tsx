"use client";

import Link from "next/link";
import { useState } from "react";
import { Button } from "~/components/ui/button";

interface Prediction {
  class: string;
  confidence: number;
}

interface LayerData {
  shape: number[];
  values: number[][];
}

interface VisualizationData {
  [layerName: string]: LayerData;
}

interface WaveformData {
  values: number[];
  sample_rate: number;
  duration: number;
}

interface ApiResponse {
  predictions: Prediction[];
  visualization: VisualizationData;
  input_spectrogram: LayerData;
  waveform: WaveformData;
}

export default function HomePage() {
  const[vizData, setVizData] = useState<null>(null);
  const [isLoading, setIsLoading] = useState(false);  // we create a useState to track if w uploading or analysing the file or not
  const[fileName, setFileName] = useState("");
  const[error, seterror] = useState<string | null>(null);

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setIsLoading(true)

  };

  return (
    <main className="min-h-screen bg-stone-50 p-8">
      <div className="mx-auto max-w-[60%]">
        <div className="mb-12 text-center">
          <h1 className="mb-4 text-4xl font-bold tracking-tight text-stone-900">AUDIO CNN VISUALIZER
            <p className="text-md mb-8 text-lg text-stone-600">Upload a WAV file to see the model's feature maps and predictions.


              <div className="flex flex-col items-center">
                <div className="relative inline-block"></div>
                  <input 
                    type="file" 
                    accept=".wav" 
                    id="file-upload"
                    disabled={isLoading}
                    className="absolute inset-0 w-full cursor-pointer opacity-0"
                  />
                  <Button 
                    disabled={isLoading}
                    className="border-stone-300" 
                    variant="outline" 
                    size="lg"
                  >
                    {isLoading ? "Analysing.." : "Choose WAV File"}
                  </Button>
              </div>
            </p>
          </h1>
        </div>
      </div>
    </main>
  );
}
