/**
 * LokLens AI API service
 * All communication with the FastAPI backend goes through this module.
 */

import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const api = axios.create({
  baseURL: `${BASE_URL}/api`,
  timeout: 60_000,
  headers: { "Accept": "application/json" },
});

// ── Types ──────────────────────────────────────────────────────────────────────

export interface Submission {
  id: string;
  input_type: "text" | "image" | "multimodal";
  status: "queued" | "processing" | "complete" | "error";
  raw_text?: string;
  created_at: string;
  completed_at?: string;
  error_message?: string;
}

export interface Claim {
  id: string;
  submission_id: string;
  claim_text: string;
  subject?: string;
  predicate?: string;
  object?: string;
  entities_json?: Record<string, string[]>;
  keywords_json?: string[];
  search_queries_json?: string[];
  created_at: string;
}

export interface Evidence {
  id: string;
  claim_id: string;
  document_id: string;
  evidence_text: string;
  support_score: number;
  contradiction_score: number;
  relevance_score: number;
  retrieval_method: string;
  contradiction_indicators_json?: Record<string, unknown>;
  created_at: string;
}

export interface Source {
  id: string;
  url: string;
  domain: string;
  source_type: string;
  authority_score: number;
  quality_score: number;
  quality_components_json?: Record<string, number>;
}

export interface ImageForensics {
  id: string;
  image_id: string;
  ela_score?: number;
  noise_anomaly?: number;
  edge_anomaly?: number;
  frequency_anomaly?: number;
  compression_anomaly?: number;
  metadata_anomaly?: number;
  copy_move_score?: number;
  ai_likelihood_score?: number;
  ai_likelihood_label?: string;
  exif_json?: Record<string, unknown>;
  analyzed_at: string;
}

export interface FullReport {
  submission_id: string;
  overall_status?: string;
  claim_verdict?: string;
  claim_confidence?: number;
  image_verdict?: string;
  image_confidence?: number;
  summary?: string;
  claims: Claim[];
  supporting_evidence: Evidence[];
  contradicting_evidence: Evidence[];
  sources: Source[];
  independent_source_count?: number;
  duplicate_source_group_count?: number;
  timeline: Array<{ id: string; event_date?: string; event_description: string }>;
  limitations: string[];
  score_components?: Record<string, unknown>;
}

export interface GraphData {
  nodes: Array<{ node_type: string; node_id: string; label: string }>;
  edges: Array<{
    from_node_type: string; from_node_id: string;
    to_node_type: string; to_node_id: string;
    relationship_type: string; weight: number;
  }>;
}

// ── API calls ──────────────────────────────────────────────────────────────────

export async function createSubmission(
  text?: string,
  imageFile?: File
): Promise<Submission> {
  const form = new FormData();
  if (text) form.append("text", text);
  if (imageFile) form.append("image", imageFile);
  const { data } = await api.post<Submission>("/submissions", form);
  return data;
}

export async function triggerAnalysis(id: string): Promise<Submission> {
  const { data } = await api.post<Submission>(`/submissions/${id}/analyze`);
  return data;
}

export async function getSubmission(id: string): Promise<Submission> {
  const { data } = await api.get<Submission>(`/submissions/${id}`);
  return data;
}

export async function getReport(id: string): Promise<FullReport> {
  const { data } = await api.get<FullReport>(`/submissions/${id}/report`);
  return data;
}

export async function getClaims(id: string): Promise<Claim[]> {
  const { data } = await api.get<Claim[]>(`/submissions/${id}/claims`);
  return data;
}

export async function getEvidence(id: string): Promise<Evidence[]> {
  const { data } = await api.get<Evidence[]>(`/submissions/${id}/evidence`);
  return data;
}

export async function getSources(id: string): Promise<Source[]> {
  const { data } = await api.get<Source[]>(`/submissions/${id}/sources`);
  return data;
}

export async function getGraph(id: string): Promise<GraphData> {
  const { data } = await api.get<GraphData>(`/submissions/${id}/graph`);
  return data;
}

export async function getApiStatus(): Promise<Record<string, unknown>> {
  const { data } = await api.get("/status");
  return data;
}
