import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AuthService } from './auth.service';

@Injectable({
  providedIn: 'root'
})
export class ChatServiceService {
  private apiUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient, private authService: AuthService) {}

  askLlama(question: string, format: string = 'text'): Observable<any> {
    const currentUser = this.authService.getCurrentUser();
    const payload = {
      question: question,
      userEmail: currentUser?.email || '',
      userRoles: currentUser?.roles || [],
      accessibleTables: currentUser?.accessibleTables || [],
      format: format
    };
    return this.http.post<any>(`${this.apiUrl}/ask-llama`, payload);
  }

  getConversationalResponse(prompt: string): Observable<any> {
    const currentUser = this.authService.getCurrentUser();
    const payload = {
      prompt: prompt,
      userEmail: currentUser?.email || '',
      userRoles: currentUser?.roles || []
    };
    return this.http.post<any>(`${this.apiUrl}/conversational-response`, payload);
  }
  
  classifyMessage(message: string, accessibleTables: string[]): Observable<any> {
    const currentUser = this.authService.getCurrentUser();
    const payload = {
      message: message,
      accessibleTables: accessibleTables,
      userEmail: currentUser?.email || '',
      userRoles: currentUser?.roles || []
    };
    return this.http.post<any>(`${this.apiUrl}/classify-message`, payload);
  }

  getAllTables(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/tables`);
  }

  getTableSchema(tableName: string): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/table-schema/${tableName}`);
  }

  askFromDocs(question: string): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/ask-docs`, { question });
  }
  
}