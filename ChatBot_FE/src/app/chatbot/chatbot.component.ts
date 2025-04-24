// chatbot.component.ts 
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { Component, ElementRef, ViewChild, AfterViewInit, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatServiceService } from '../chat-service.service';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { AuthService, User, UserRole } from '../auth.service';
import { Router } from '@angular/router';

interface Message {
  name: string;
  message: string;
  isTable?: boolean;
}

interface TableInfo {
  name: string;
  schema?: string;
}

@Component({
  selector: 'app-chatbot',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './chatbot.component.html',
  styleUrl: './chatbot.component.css',
  providers: []
})
export class ChatbotComponent implements AfterViewInit, OnInit {
  @ViewChild('messageInput') messageInput!: ElementRef<HTMLInputElement>;
  @ViewChild('chatMessages') chatMessages!: ElementRef<HTMLDivElement>;
  
  messages: Message[] = [];
  inputMessage: string = '';
  isOpen: boolean = false;
  isLoading: boolean = false;
  currentUser: User | null = null;
  accessibleTables: string[] = [];
  tableSchemas: Map<string, string> = new Map();
  selectedTable: string = '';
  
  constructor(
    private chatService: ChatServiceService,
    private sanitizer: DomSanitizer,
    private authService: AuthService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.authService.currentUser.subscribe(user => {
      if (user) {
        this.currentUser = user;
        this.accessibleTables = user.accessibleTables;
        const roleName = user.roles.length > 0 ? user.roles[0].name : 'User';
        
        // Add welcome message
        this.messages.push({
          name: 'ArcKa',
          message: `Welcome ${roleName}! I’m ArcKa — your thinking partner. Ask me anything, and I’ll help you figure it out. How can I help you today?`
        });
        
        // Set the default table if available
        if (this.accessibleTables.length > 0) {
          this.selectedTable = this.accessibleTables[0];
        }
      } else {
        this.router.navigate(['/login']);
      }
    });
  }

  ngAfterViewInit(): void {
    // Only focus the input if the chatbox is open
    if (this.messageInput && this.isOpen) {
      this.messageInput.nativeElement.focus();
    }
  }

  toggleChatbox(): void {
    this.isOpen = !this.isOpen;
    console.log("Chatbox toggled: ", this.isOpen);
    
    if (this.isOpen && this.messageInput) {
      setTimeout(() => {
        this.messageInput.nativeElement.focus();
        this.scrollToBottom(); 
      }, 100);
    }
  }

  handleKeyUp(event: KeyboardEvent): void {
    if (event.key === 'Enter') {
      this.sendMessage();
    }
  }
  
  logout(): void {
    this.authService.logout();
    this.router.navigate(['/login']);
  }
  
  sanitizeHtml(html: string): SafeHtml {
    return this.sanitizer.bypassSecurityTrustHtml(html);
  }

//   sendMessage(): void {
//     if (this.inputMessage.trim() === '' || this.isLoading) {
//       return;
//     }

//     const userMessage = this.inputMessage;
//     this.messages.push({ name: 'User', message: userMessage });
//     this.inputMessage = '';
    
//     this.scrollToBottom();
//     this.isLoading = true;
    
//     // First, classify the message using the LLM to determine if it's conversational or a database query
//     this.chatService.classifyMessage(userMessage, this.accessibleTables).subscribe({
//       next: (classification) => {
//         if (classification.success && classification.type === 'conversational') {
//           // Handle as conversational message
//           const prompt = `The user said: "${userMessage}"
          
// You are a helpful database assistant named ArcKa. Respond in a friendly, conversational way.
// Keep your response brief and natural.`;
          
//           this.chatService.getConversationalResponse(prompt).subscribe({
//             next: (response) => {
//               this.isLoading = false;
//               if (response.success) {
//                 this.messages.push({ 
//                   name: 'ArcKa', 
//                   message: response.message
//                 });
//               } else {
//                 this.messages.push({ 
//                   name: 'ArcKa', 
//                   message: "I'm sorry, I couldn't process your message. How can I help you with your data queries?"
//                 });
//               }
//               this.scrollToBottom();
//             },
//             error: (error) => {
//               this.handleResponseError(error);
//             }
//           });
//         } else {
//           // Handle as database query
//           this.handleDatabaseQuery(userMessage);
//         }
//       },
//       error: (error) => {
//         // If classification fails, default to treating it as a database query
//         console.error('Classification error:', error);
//         this.handleDatabaseQuery(userMessage);
//       }
//     });
//   }
sendMessage(): void {
  if (this.inputMessage.trim() === '' || this.isLoading) {
    return;
  }

  const userMessage = this.inputMessage;
  this.messages.push({ name: 'User', message: userMessage });
  this.inputMessage = '';

  this.scrollToBottom();
  this.isLoading = true;

  this.chatService.classifyMessage(userMessage, this.accessibleTables).subscribe({
    next: (classification) => {
      const type = classification?.type || 'database_query';

      if (type === 'conversational') {
        const prompt = `The user said: "${userMessage}"

You are a helpful assistant named ArcKa. Respond in a friendly, conversational way.
Keep your response brief and natural.`;

        this.chatService.getConversationalResponse(prompt).subscribe({
          next: (response) => {
            this.isLoading = false;
            if (response.success) {
              this.messages.push({ name: 'ArcKa', message: response.message });
            } else {
              this.messages.push({
                name: 'ArcKa',
                message: "I'm sorry, I couldn't process your message. How can I help you?"
              });
            }
            this.scrollToBottom();
          },
          error: (error) => {
            this.handleResponseError(error);
          }
        });

      } else if (type === 'document_query') {
        this.chatService.askFromDocs(userMessage).subscribe({
          next: (response) => {
            this.isLoading = false;
            if (response.success) {
              this.messages.push({
                name: 'ArcKa',
                message: response.message
              });
            } else {
              this.messages.push({
                name: 'ArcKa',
                message: response.error || "I couldn't find anything relevant in the documents."
              });
            }
            this.scrollToBottom();
          },
          error: (error) => {
            this.handleResponseError(error);
          }
        });

      } else {
        // default to database_query
        this.chatService.askLlama(userMessage, "table").subscribe({
          next: (response) => {
            this.isLoading = false;
            if (response.success && response.format === "table" && response.results?.length) {
              const tableHtml = this.formatResultsAsTable(response.results);
              this.messages.push({
                name: 'ArcKa',
                message: tableHtml,
                isTable: true
              });
            } else {
              this.messages.push({
                name: 'ArcKa',
                message: response.message || "No data found for your query."
              });
            }
            this.scrollToBottom();
          },
          error: (error) => {
            this.handleResponseError(error);
          }
        });
      }
    },
    error: (error) => {
      console.error('Classification error:', error);
      this.handleDatabaseQuery(userMessage); // fallback
    }
  });
}

  handleDatabaseQuery(userMessage: string): void {
    const formatPreference = userMessage.toLowerCase().includes("tabular format") || 
                           userMessage.toLowerCase().includes("table format") ? 
                           "table" : "text";
    
    // We no longer need to manually detect the table or specify a default table
    // The backend will handle table detection based on the question and available metadata
    
    this.chatService.askLlama(userMessage, formatPreference).subscribe({
      next: (response) => {
        this.isLoading = false;
        
        if (response.success) {
          if (response.format === "table" && response.results && response.results.length > 0) {
            // If the response is in table format
            let tableHtml = this.formatResultsAsTable(response.results);
            this.messages.push({ 
              name: 'ArcKa', 
              message: tableHtml,
              isTable: true
            });
          } else if (response.message) {
            // If the response is in natural language text format
            this.messages.push({ 
              name: 'ArcKa', 
              message: response.message
            });
          } else {
            // Fallback message if no results or message
            this.messages.push({ 
              name: 'ArcKa', 
              message: 'No results found for your query.'
            });
          }
        } else {
          // Handle error cases
          let errorMessage = '';
          
          // Check if it's a permissions issue
          if (response.error && response.error.includes('permission')) {
            errorMessage = `Sorry, you don't have permission to access this data. As a ${this.currentUser?.roles[0]?.name || 'User'}, you can only query certain tables.`;
          } 
          // Check if it's a "specify table" error
          else if (response.error && (response.error.includes('specify a table') || response.error.includes('determine which table'))) {
            const examples = this.accessibleTables.slice(0, 3).join(', ');
            errorMessage = `I couldn't determine which table to query based on your question. Could you try rephrasing your question with more specific details? You have access to tables like ${examples}${this.accessibleTables.length > 3 ? '...' : ''}.`;
          } 
          else {
            errorMessage = "I'm sorry, I couldn't process your request. Please try asking your question in a different way.";
          }
          
          console.error('Error:', response.error);
          this.messages.push({ name: 'ArcKa', message: errorMessage });
        }
        
        this.scrollToBottom();
      },
      error: (error) => {
        this.handleResponseError(error);
      }
    });
  }
  
  handleResponseError(error: any): void {
    this.isLoading = false;
    console.error('Error:', error);
    this.messages.push({ 
      name: 'ArcKa', 
      message: "Sorry, I couldn't process your request. Please try again." 
    });
    this.scrollToBottom();
  }

  formatResultsAsTable(results: any[]): string {
    if (!Array.isArray(results) || results.length === 0) {
      return 'No results found.';
    }
    
    const keys = Object.keys(results[0]);
    let tableHtml = '<table class="sql-table">';
    tableHtml += '<tr>';
    keys.forEach(key => {
      tableHtml += `<th>${key}</th>`;
    });
    tableHtml += '</tr>';
    results.forEach(row => {
      tableHtml += '<tr>';
      keys.forEach(key => {
        tableHtml += `<td>${row[key] !== null ? row[key] : ''}</td>`;
      });
      tableHtml += '</tr>';
    });
    
    tableHtml += '</table>';
    return tableHtml;
  }

  scrollToBottom(): void {
    setTimeout(() => {
      if (this.chatMessages) {
        const element = this.chatMessages.nativeElement;
        element.scrollTop = element.scrollHeight;
      }
    }, 0);
  }
}